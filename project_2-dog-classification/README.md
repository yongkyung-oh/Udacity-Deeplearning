
# Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App 

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully! 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the Jupyter Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this Jupyter notebook.



---
### Why We're Here 

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!). 

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Write your Algorithm
* [Step 6](#step6): Test Your Algorithm

---
<a id='step0'></a>
## Step 0: Import Datasets

Make sure that you've downloaded the required human and dog datasets:
* Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in this project's home directory, at the location `/dog_images`. 

* Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the home directory, at location `/lfw`.  

*Note: If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder.*

In the code cell below, we save the file paths for both the human (LFW) dataset and dog dataset in the numpy arrays `human_files` and `dog_files`.


```python
# Install somd additional packages to use
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install tabulate
```

    Looking in indexes: http://ftp.daumkakao.com/pypi/simple
    Requirement already satisfied: tabulate in /home/yongkyung/anaconda3/envs/tf-gpu/lib/python3.6/site-packages (0.8.3)



```python
import os 
import numpy as np
from functools import partial
from tabulate import tabulate
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("{}/data/lfw/*/*".format(os.getcwd())))
dog_files = np.array(glob("{}/data/dog_images/*/*/*".format(os.getcwd())))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
```

    There are 13233 total human images.
    There are 8351 total dog images.


<a id='step1'></a>
## Step 1: Detect Humans

In this section, we use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  

OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.  In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])

# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1



![png](output_4_1.png)



```python
img = cv2.imread(human_files[1])

# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1



![png](output_5_1.png)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face? 

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ See the output below.


```python
from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]
#-#-# Do NOT modify the code above this line. #-#-#

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
human_count = 0
for i in range(0, 100):
    if face_detector(human_files_short[i]):
        human_count += 1
dog_count = 0
for i in range(0, 100):
    if face_detector(dog_files_short[i]):
        dog_count += 1

print("{} of human face detected out of 100 samples.".format(human_count))
print("{} of dog face detected out of 100 samples.".format(dog_count))
```

    100 of human face detected out of 100 samples.
    19 of dog face detected out of 100 samples.


I wrote my own code and implemented some parts from following reference: https://necromuralist.github.io/In-Too-Deep/posts/nano/dog-breed-classifier/dog-app/


```python
def species_scorer(predictor, true_species, false_species, labels):
    misses = [predictor(str(image)) for image in false_species]
    false_positives = sum(misses)
    true_positives = sum([predictor(str(image)) for image in true_species])
    false_negatives = len(true_species) - true_positives
    others = len(false_species)
    expected = len(true_species)
    values = ("{:.2f}%".format(100 * true_positives/expected),
            "{:.2f}%".format(100 * false_positives/others),
              "{:.2f}".format((2 * true_positives)/(2 * true_positives
                                                    + false_positives
                                                    + false_negatives)))
    table = zip(labels, values)
    print(tabulate(table, tablefmt="github", headers=["Metric", "Value"]))
    return misses
```


```python
face_scorer = partial(species_scorer,
                      true_species=human_files_short,
                      false_species=dog_files_short,
                      labels=("First 100 images in `human_files` detected with a face",
                              "First 100 images in `dog_files` detected with a face",
                              "F1"))
```


```python
open_cv_false_positives = face_scorer(face_detector)
```

    | Metric                                                 | Value   |
    |--------------------------------------------------------|---------|
    | First 100 images in `human_files` detected with a face | 100.00% |
    | First 100 images in `dog_files` detected with a face   | 19.00%  |
    | F1                                                     | 0.91    |


---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a [pre-trained model](http://pytorch.org/docs/master/torchvision/models.html) to detect dogs in images.  

### Obtain Pre-trained VGG-16 Model

The code cell below downloads the VGG-16 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  

Due to the memory issue, I cleared memory and setup again.


```python
import os 
import numpy as np
from functools import partial
from tabulate import tabulate
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("{}/data/lfw/*/*".format(os.getcwd())))
dog_files = np.array(glob("{}/data/dog_images/*/*/*".format(os.getcwd())))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
```

    There are 13233 total human images.
    There are 8351 total dog images.



```python
from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]
```


```python
import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()
```


```python
device = torch.device("cuda" if use_cuda else "cpu")
print("Using {}".format(device))
```

    Using cuda


Given an image, this pre-trained VGG-16 model returns a prediction (derived from the 1000 possible categories in ImageNet) for the object that is contained in the image.

### (IMPLEMENTATION) Making Predictions with a Pre-trained Model

In the next code cell, you will write a function that accepts a path to an image (such as `'dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg'`) as input and returns the index corresponding to the ImageNet class that is predicted by the pre-trained VGG-16 model.  The output should always be an integer between 0 and 999, inclusive.

Before writing the function, make sure that you take the time to learn  how to appropriately pre-process tensors for pre-trained models in the [PyTorch documentation](http://pytorch.org/docs/stable/torchvision/models.html).


```python
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

IMAGE_SIZE = 224
IMAGE_HALF_SIZE = IMAGE_SIZE//2

vgg_transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
```


```python
def model_predict(image_path: str, model: nn.Module, transform: transforms.Compose):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    
    image = Image.open(str(image_path))
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    probabilities = torch.exp(output)
    _, top_class = probabilities.topk(1, dim=1)
    return top_class.item()   

VGG16_predict = partial(model_predict, model=VGG16, transform=vgg_transform)
```

### (IMPLEMENTATION) Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained VGG-16 model, we need only check if the pre-trained model predicts an index between 151 and 268 (inclusive).

Use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).

According to the image net dictionay, `'Chihuahua'=151` and `'Mexican hairless'=268` are specified. 


```python
DOG_LOWER, DOG_UPPER = 150, 268
```


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path, predictor: callable=VGG16_predict):
    ## TODO: Complete the function.
    
    return DOG_LOWER < VGG16_predict(img_path) < DOG_UPPER
```

### (IMPLEMENTATION) Assess the Dog Detector

__Question 2:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__ See the output below.



```python
def species_scorer(predictor, true_species, false_species, labels):
    misses = [predictor(str(image)) for image in false_species]
    false_positives = sum(misses)
    true_positives = sum([predictor(str(image)) for image in true_species])
    false_negatives = len(true_species) - true_positives
    others = len(false_species)
    expected = len(true_species)
    values = ("{:.2f}%".format(100 * true_positives/expected),
            "{:.2f}%".format(100 * false_positives/others),
              "{:.2f}".format((2 * true_positives)/(2 * true_positives
                                                    + false_positives
                                                    + false_negatives)))
    table = zip(labels, values)
    print(tabulate(table, tablefmt="github", headers=["Metric", "Value"]))
    return misses
```


```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

dog_scorer = partial(species_scorer,
                     true_species=dog_files_short,
                     false_species=human_files_short,
                     labels=("Images in `dog_files_short` with a detected dog",
                             "Images in `human_files_short with a detected dog", "F1"))
```


```python
VGG_false_dogs = dog_scorer(dog_detector)
```

    | Metric                                           | Value   |
    |--------------------------------------------------|---------|
    | Images in `dog_files_short` with a detected dog  | 100.00% |
    | Images in `human_files_short with a detected dog | 2.00%   |
    | F1                                               | 0.99    |


We suggest VGG-16 as a potential network to detect dog images in your algorithm, but you are free to explore other pre-trained networks (such as [Inception-v3](http://pytorch.org/docs/master/torchvision/models.html#inception-v3), [ResNet-50](http://pytorch.org/docs/master/torchvision/models.html#id3), etc).  Please use the code cell below to test other pre-trained PyTorch models.  If you decide to pursue this _optional_ task, report performance on `human_files_short` and `dog_files_short`.

## Implementation using Inception


```python
import torch
import torchvision.models as models

inception = models.inception_v3(pretrained=True)
inception.eval()

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    inception = inception.cuda()
```


```python
device = torch.device("cuda" if use_cuda else "cpu")
print("Using {}".format(device))
```

    Using cuda



```python
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

INCEPTION_IMAGE_SIZE = 299

inception_transforms = transforms.Compose([transforms.Resize(INCEPTION_IMAGE_SIZE),
                                           transforms.CenterCrop(INCEPTION_IMAGE_SIZE),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

inception_predict = partial(model_predict, model=inception, transform=inception_transforms)
```


```python
inception_dog_detector = partial(dog_detector, predictor=inception_predict)
```


```python
inception_false_dogs = dog_scorer(inception_dog_detector)
```

    | Metric                                           | Value   |
    |--------------------------------------------------|---------|
    | Images in `dog_files_short` with a detected dog  | 100.00% |
    | Images in `human_files_short with a detected dog | 1.00%   |
    | F1                                               | 1.00    |


## Implementation using ResNet50

Due to the memory issue, I cleared memory and setup again.


```python
import os 
import numpy as np
from functools import partial
from tabulate import tabulate
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("{}/data/lfw/*/*".format(os.getcwd())))
dog_files = np.array(glob("{}/data/dog_images/*/*/*".format(os.getcwd())))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
```

    There are 13233 total human images.
    There are 8351 total dog images.



```python
from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]
```

I wrote my own code and implemented some parts from following reference: http://kitmarks.com/dog_app.html


```python
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```

    Using TensorFlow backend.



```python
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```


```python
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet_predict(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```

According to the image net dictionay, `'Chihuahua'=151` and `'Mexican hairless'=268` are specified. 


```python
DOG_LOWER, DOG_UPPER = 150, 268
```


```python
def Resnet_dog_detector(img_path):
    return DOG_LOWER < ResNet_predict(img_path) < DOG_UPPER
```


```python
def species_scorer(predictor, true_species, false_species, labels):
    misses = [predictor(str(image)) for image in false_species]
    false_positives = sum(misses)
    true_positives = sum([predictor(str(image)) for image in true_species])
    false_negatives = len(true_species) - true_positives
    others = len(false_species)
    expected = len(true_species)
    values = ("{:.2f}%".format(100 * true_positives/expected),
            "{:.2f}%".format(100 * false_positives/others),
              "{:.2f}".format((2 * true_positives)/(2 * true_positives
                                                    + false_positives
                                                    + false_negatives)))
    table = zip(labels, values)
    print(tabulate(table, tablefmt="github", headers=["Metric", "Value"]))
    return misses
```


```python
dog_scorer = partial(species_scorer,
                     true_species=dog_files_short,
                     false_species=human_files_short,
                     labels=("Images in `dog_files_short` with a detected dog",
                             "Images in `human_files_short with a detected dog", "F1"))
```


```python
ResNet50_false_dogs = dog_scorer(Resnet_dog_detector)
```

    | Metric                                           | Value   |
    |--------------------------------------------------|---------|
    | Images in `dog_files_short` with a detected dog  | 97.00%  |
    | Images in `human_files_short with a detected dog | 1.00%   |
    | F1                                               | 0.98    |


---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 10%.  In Step 4 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have trouble distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | - 
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun!

### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset

Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dog_images/train`, `dog_images/valid`, and `dog_images/test`, respectively).  You may find [this documentation on custom datasets](http://pytorch.org/docs/stable/torchvision/datasets.html) to be a useful resource.  If you are interested in augmenting your training and/or validation data, check out the wide variety of [transforms](http://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transform)!

Due to the memory issue, I cleared memory and setup again.


```python
# Install somd additional packages to use
# Install a pip package in the current Jupyter kernel
import sys
!{sys.executable} -m pip install --upgrade pip
!{sys.executable} -m pip install tabulate
```

    Looking in indexes: http://ftp.daumkakao.com/pypi/simple
    Requirement already up-to-date: pip in /home/yongkyung/anaconda3/envs/tf-gpu/lib/python3.6/site-packages (19.0.3)
    Looking in indexes: http://ftp.daumkakao.com/pypi/simple
    Requirement already satisfied: tabulate in /home/yongkyung/anaconda3/envs/tf-gpu/lib/python3.6/site-packages (0.8.3)



```python
import os 
import numpy as np
from functools import partial
from tabulate import tabulate
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("{}/data/lfw/*/*".format(os.getcwd())))
dog_files = np.array(glob("{}/data/dog_images/*/*/*".format(os.getcwd())))
test_files = np.array(glob("{}/data/test/*".format(os.getcwd())))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
print('There are %d total test images.' % len(test_files))
```

    There are 13233 total human images.
    There are 8351 total dog images.
    There are 6 total test images.



```python
from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]
```


```python
import os
import torch
from torchvision import datasets

from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes

SCRATCH_IMAGE_SIZE = 224

train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(SCRATCH_IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize(350),
                                     transforms.CenterCrop(SCRATCH_IMAGE_SIZE),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
```


```python
from datetime import datetime
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

I wrote my own code and implemented some parts from following reference: https://necromuralist.github.io/In-Too-Deep/posts/nano/dog-breed-classifier/dog-app/

Specify the path of dataset and data loaders for the dog dataset


```python
path = os.getcwd()  
print ("The current working directory is %s" % path)  
```

    The current working directory is /HDD/yongkyung/project-dog-classification



```python
from pathlib import Path

DOG_PATH = Path("{}/data/dog_images".format(os.getcwd()))
dog_training_path = DOG_PATH.joinpath("train")
dog_validation_path = DOG_PATH.joinpath("valid")
dog_testing_path = DOG_PATH.joinpath("test")
```


```python
training = datasets.ImageFolder(root=str(dog_training_path),
                                transform=train_transform)
validation = datasets.ImageFolder(root=str(dog_validation_path),
                                  transform=test_transform)
testing = datasets.ImageFolder(root=str(dog_testing_path),
                               transform=test_transform)
```


```python
BATCH_SIZE = 32
WORKERS = 0

train_batches = torch.utils.data.DataLoader(training, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=WORKERS)
validation_batches = torch.utils.data.DataLoader(
    validation, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
test_batches = torch.utils.data.DataLoader(
    testing, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)

loaders_scratch = dict(train=train_batches,
                       validation=validation_batches,
                       test=test_batches)
```

**Question 3:** Describe your chosen procedure for preprocessing the data. 
- How does your code resize the images (by cropping, stretching, etc)?  What size did you pick for the input tensor, and why?
- Did you decide to augment the dataset?  If so, how (through translations, flips, rotations, etc)?  If not, why not?


**Answer**: 
* I selected the same size as inception_v3 (299 px)
* The training images are resized by cropping and testing images are resized by scaling then cropping.
* The transformation includes rotation, cropping and horizontal flipping.

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  Use the template in the code cell below.


```python
BREEDS = len(training.classes)
print("There are {} breeds.".format(BREEDS))
```

    There are 133 breeds.



```python
import torch.nn as nn
import torch.nn.functional as F

LAYER_ONE_IN = 3
LAYER_ONE_OUT = 16
LAYER_TWO_OUT = LAYER_ONE_OUT * 2 # 32
LAYER_THREE_OUT = LAYER_TWO_OUT * 2 # 64
LAYER_FOUR_OUT = LAYER_THREE_OUT * 2 # 128
FLATTEN_TO = (SCRATCH_IMAGE_SIZE//16)**2 * LAYER_FOUR_OUT 
FULLY_CONNECTED_OUT = int(str(FLATTEN_TO)[:3])//100 * 100
KERNEL = 3
PADDING = 1
```


```python
# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(LAYER_ONE_IN, LAYER_ONE_OUT,
                               KERNEL, padding=PADDING)
        self.conv2 = nn.Conv2d(LAYER_ONE_OUT, LAYER_TWO_OUT,
                               KERNEL, padding=PADDING)
        self.conv3 = nn.Conv2d(LAYER_TWO_OUT, LAYER_THREE_OUT,
                               KERNEL, padding=PADDING)
        self.conv4 = nn.Conv2d(LAYER_THREE_OUT, LAYER_FOUR_OUT,
                               KERNEL, padding=PADDING)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer
        self.fc1 = nn.Linear(FLATTEN_TO, FULLY_CONNECTED_OUT)
        self.fc2 = nn.Linear(FULLY_CONNECTED_OUT, BREEDS)
        # dropout layer
        self.dropout = nn.Dropout(0.25)
        return
    
    ## Define layers of a CNN
    def forward(self, x):
        ## Define forward behavior
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = x.view(-1, FLATTEN_TO)
        x = self.dropout(x)

        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
```

__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  

__Answer:__ 

I tried several CNN strucure, but most of them showed poor result or unexpected error. 
* 4 convolution layers, which doule the depth while halving the height and width (using MaxPool)
* Flatten the layer with 300 output
* To reduce the likelihood of overfitting, model is applied to dropout the activation layers
* Applied ReLU activation to make the model nonlinear

* Key idea from following reference: https://necromuralist.github.io/In-Too-Deep/posts/nano/dog-breed-classifier/dog-app/

### (IMPLEMENTATION) Specify Loss Function and Optimizer

Use the next code cell to specify a [loss function](http://pytorch.org/docs/stable/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/stable/optim.html).  Save the chosen loss function as `criterion_scratch`, and the optimizer as `optimizer_scratch` below.


```python
import torch.optim as optimizer

### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optimizer.SGD(model_scratch.parameters(), lr=0.001, momentum=0.9)
```

### (IMPLEMENTATION) Train and Validate the Model

Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_scratch.pt'`.


```python
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path,
          print_function: callable=print,
          is_inception: bool=False):

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    
    # check the keys are right so you don't waste an entire epoch to find out
    training_batches = loaders["train"]
    validation_batches = loaders["validation"]
    started = datetime.now()
    print_function("Training Started: {}".format(started))
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        epoch_started = datetime.now()
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in training_batches:
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss /= len(training_batches.dataset)

        ######################    
        # validate the model #
        ######################
        model.eval()
        for data, target in validation_batches:
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)
        valid_loss /= len(validation_batches.dataset)
        print_function('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\tElapsed: {}'.format(
            epoch,                     
            train_loss,
            valid_loss,
            datetime.now() - epoch_started,
            ))
        
        if valid_loss < valid_loss_min:
            print_function(
                ("Validation loss decreased ({:.6f} --> {:.6f}). "
                 "Saving model ...").format(
                     valid_loss_min,
                     valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
    ended = datetime.now()
    print_function("Training Ended: {}".format(ended))
    print_function("Total Training Time: {}".format(ended - started))            
    return model
```


```python
class Tee:
    """Save the input to a file and print it

    Args:
     log_name: name to give the log    
     directory_path: path to the directory for the file
    """
    def __init__(self, log_name: str, 
                 directory_name: str="/") -> None:
        self.directory_name = directory_name
        self.log_name = log_name
        self._path = None
        self._log = None
        return

    @property
    def path(self) -> Path:
        """path to the log-file"""
        if self._path is None:
            self._path = Path(self.directory_name).expanduser()
            assert self._path.is_dir()
            self._path = self._path.joinpath(self.log_name)
        return self._path

    @property
    def log(self):
        """File object to write log to"""
        if self._log is None:
            self._log = self.path.open("w", buffering=1)
        return self._log

    def __call__(self, line: str) -> None:
        """Writes to the file and stdout

        Args:
         line: text to emit
        """
        self.log.write("{}\n".format(line))
        print(line)
        return
```


```python
MODEL_PATH = Path(os.getcwd())
scratch_path = MODEL_PATH.joinpath("model_scratch.pt")
scratch_log = Tee(log_name="scratch_train.log", directory_name=MODEL_PATH)
```


```python
EPOCHS = 100
```


```python
model_scratch = train(EPOCHS, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, scratch_path, print_function=print)
```

    Training Started: 2019-04-01 07:45:58.783654
    Epoch: 1 	Training Loss: 4.888101 	Validation Loss: 4.884808	Elapsed: 0:01:15.319154
    Validation loss decreased (inf --> 4.884808). Saving model ...
    Epoch: 2 	Training Loss: 4.882133 	Validation Loss: 4.876177	Elapsed: 0:01:14.177591
    Validation loss decreased (4.884808 --> 4.876177). Saving model ...
    Epoch: 3 	Training Loss: 4.874599 	Validation Loss: 4.863225	Elapsed: 0:01:14.410069
    Validation loss decreased (4.876177 --> 4.863225). Saving model ...
    Epoch: 4 	Training Loss: 4.863587 	Validation Loss: 4.841865	Elapsed: 0:01:14.988043
    Validation loss decreased (4.863225 --> 4.841865). Saving model ...
    Epoch: 5 	Training Loss: 4.838524 	Validation Loss: 4.796649	Elapsed: 0:01:15.416339
    Validation loss decreased (4.841865 --> 4.796649). Saving model ...
    Epoch: 6 	Training Loss: 4.813162 	Validation Loss: 4.759795	Elapsed: 0:01:15.220287
    Validation loss decreased (4.796649 --> 4.759795). Saving model ...
    Epoch: 7 	Training Loss: 4.782507 	Validation Loss: 4.717613	Elapsed: 0:01:15.039709
    Validation loss decreased (4.759795 --> 4.717613). Saving model ...
    Epoch: 8 	Training Loss: 4.762538 	Validation Loss: 4.698938	Elapsed: 0:01:15.290507
    Validation loss decreased (4.717613 --> 4.698938). Saving model ...
    Epoch: 9 	Training Loss: 4.748712 	Validation Loss: 4.683372	Elapsed: 0:01:15.517570
    Validation loss decreased (4.698938 --> 4.683372). Saving model ...
    Epoch: 10 	Training Loss: 4.725505 	Validation Loss: 4.680891	Elapsed: 0:01:14.922617
    Validation loss decreased (4.683372 --> 4.680891). Saving model ...
    Epoch: 11 	Training Loss: 4.712450 	Validation Loss: 4.642180	Elapsed: 0:01:14.860153
    Validation loss decreased (4.680891 --> 4.642180). Saving model ...
    Epoch: 12 	Training Loss: 4.676686 	Validation Loss: 4.590708	Elapsed: 0:01:14.812883
    Validation loss decreased (4.642180 --> 4.590708). Saving model ...
    Epoch: 13 	Training Loss: 4.631384 	Validation Loss: 4.515303	Elapsed: 0:01:15.019521
    Validation loss decreased (4.590708 --> 4.515303). Saving model ...
    Epoch: 14 	Training Loss: 4.583804 	Validation Loss: 4.459832	Elapsed: 0:01:15.439231
    Validation loss decreased (4.515303 --> 4.459832). Saving model ...
    Epoch: 15 	Training Loss: 4.565014 	Validation Loss: 4.441948	Elapsed: 0:01:15.401808
    Validation loss decreased (4.459832 --> 4.441948). Saving model ...
    Epoch: 16 	Training Loss: 4.540208 	Validation Loss: 4.434919	Elapsed: 0:01:15.269701
    Validation loss decreased (4.441948 --> 4.434919). Saving model ...
    Epoch: 17 	Training Loss: 4.525647 	Validation Loss: 4.428009	Elapsed: 0:01:15.457640
    Validation loss decreased (4.434919 --> 4.428009). Saving model ...
    Epoch: 18 	Training Loss: 4.495089 	Validation Loss: 4.379812	Elapsed: 0:01:15.870698
    Validation loss decreased (4.428009 --> 4.379812). Saving model ...
    Epoch: 19 	Training Loss: 4.486332 	Validation Loss: 4.411477	Elapsed: 0:01:15.184311
    Epoch: 20 	Training Loss: 4.471517 	Validation Loss: 4.349521	Elapsed: 0:01:16.308610
    Validation loss decreased (4.379812 --> 4.349521). Saving model ...
    Epoch: 21 	Training Loss: 4.449734 	Validation Loss: 4.341425	Elapsed: 0:01:16.636765
    Validation loss decreased (4.349521 --> 4.341425). Saving model ...
    Epoch: 22 	Training Loss: 4.434515 	Validation Loss: 4.332984	Elapsed: 0:01:15.641812
    Validation loss decreased (4.341425 --> 4.332984). Saving model ...
    Epoch: 23 	Training Loss: 4.413965 	Validation Loss: 4.292701	Elapsed: 0:01:15.732877
    Validation loss decreased (4.332984 --> 4.292701). Saving model ...
    Epoch: 24 	Training Loss: 4.421793 	Validation Loss: 4.271495	Elapsed: 0:01:15.401088
    Validation loss decreased (4.292701 --> 4.271495). Saving model ...
    Epoch: 25 	Training Loss: 4.389640 	Validation Loss: 4.275419	Elapsed: 0:01:16.229068
    Epoch: 26 	Training Loss: 4.364037 	Validation Loss: 4.237111	Elapsed: 0:01:15.206343
    Validation loss decreased (4.271495 --> 4.237111). Saving model ...
    Epoch: 27 	Training Loss: 4.353119 	Validation Loss: 4.248034	Elapsed: 0:01:15.253335
    Epoch: 28 	Training Loss: 4.314281 	Validation Loss: 4.249918	Elapsed: 0:01:15.303251
    Epoch: 29 	Training Loss: 4.314610 	Validation Loss: 4.222720	Elapsed: 0:01:14.914072
    Validation loss decreased (4.237111 --> 4.222720). Saving model ...
    Epoch: 30 	Training Loss: 4.286888 	Validation Loss: 4.154797	Elapsed: 0:01:16.123790
    Validation loss decreased (4.222720 --> 4.154797). Saving model ...
    Epoch: 31 	Training Loss: 4.264790 	Validation Loss: 4.149404	Elapsed: 0:01:14.986849
    Validation loss decreased (4.154797 --> 4.149404). Saving model ...
    Epoch: 32 	Training Loss: 4.248042 	Validation Loss: 4.112811	Elapsed: 0:01:15.261638
    Validation loss decreased (4.149404 --> 4.112811). Saving model ...
    Epoch: 33 	Training Loss: 4.235003 	Validation Loss: 4.125240	Elapsed: 0:01:14.732588
    Epoch: 34 	Training Loss: 4.208198 	Validation Loss: 4.143103	Elapsed: 0:01:15.276475
    Epoch: 35 	Training Loss: 4.187578 	Validation Loss: 4.064736	Elapsed: 0:01:15.321564
    Validation loss decreased (4.112811 --> 4.064736). Saving model ...
    Epoch: 36 	Training Loss: 4.174126 	Validation Loss: 4.071590	Elapsed: 0:01:15.206057
    Epoch: 37 	Training Loss: 4.161112 	Validation Loss: 4.066501	Elapsed: 0:01:14.808307
    Epoch: 38 	Training Loss: 4.156722 	Validation Loss: 4.036955	Elapsed: 0:01:14.917136
    Validation loss decreased (4.064736 --> 4.036955). Saving model ...
    Epoch: 39 	Training Loss: 4.130237 	Validation Loss: 4.011857	Elapsed: 0:01:14.858254
    Validation loss decreased (4.036955 --> 4.011857). Saving model ...
    Epoch: 40 	Training Loss: 4.112546 	Validation Loss: 4.025993	Elapsed: 0:01:15.207602
    Epoch: 41 	Training Loss: 4.110896 	Validation Loss: 4.008568	Elapsed: 0:01:15.141228
    Validation loss decreased (4.011857 --> 4.008568). Saving model ...
    Epoch: 42 	Training Loss: 4.087590 	Validation Loss: 4.005353	Elapsed: 0:01:15.283930
    Validation loss decreased (4.008568 --> 4.005353). Saving model ...
    Epoch: 43 	Training Loss: 4.054631 	Validation Loss: 3.967872	Elapsed: 0:01:15.322364
    Validation loss decreased (4.005353 --> 3.967872). Saving model ...
    Epoch: 44 	Training Loss: 4.033874 	Validation Loss: 3.919613	Elapsed: 0:01:15.232352
    Validation loss decreased (3.967872 --> 3.919613). Saving model ...
    Epoch: 45 	Training Loss: 4.029896 	Validation Loss: 3.926750	Elapsed: 0:01:15.028084
    Epoch: 46 	Training Loss: 4.021067 	Validation Loss: 3.924347	Elapsed: 0:01:15.529552
    Epoch: 47 	Training Loss: 4.001160 	Validation Loss: 3.942044	Elapsed: 0:01:15.019863
    Epoch: 48 	Training Loss: 3.979588 	Validation Loss: 3.907110	Elapsed: 0:01:15.247436
    Validation loss decreased (3.919613 --> 3.907110). Saving model ...
    Epoch: 49 	Training Loss: 3.973529 	Validation Loss: 3.856377	Elapsed: 0:01:15.281676
    Validation loss decreased (3.907110 --> 3.856377). Saving model ...
    Epoch: 50 	Training Loss: 3.929512 	Validation Loss: 3.829980	Elapsed: 0:01:14.951201
    Validation loss decreased (3.856377 --> 3.829980). Saving model ...
    Epoch: 51 	Training Loss: 3.905064 	Validation Loss: 3.862928	Elapsed: 0:01:14.871103
    Epoch: 52 	Training Loss: 3.907402 	Validation Loss: 3.841714	Elapsed: 0:01:15.218075
    Epoch: 53 	Training Loss: 3.914395 	Validation Loss: 3.798485	Elapsed: 0:01:14.975433
    Validation loss decreased (3.829980 --> 3.798485). Saving model ...
    Epoch: 54 	Training Loss: 3.852971 	Validation Loss: 3.868392	Elapsed: 0:01:14.888520
    Epoch: 55 	Training Loss: 3.856205 	Validation Loss: 3.802480	Elapsed: 0:01:14.888174
    Epoch: 56 	Training Loss: 3.854601 	Validation Loss: 3.777117	Elapsed: 0:01:14.886755
    Validation loss decreased (3.798485 --> 3.777117). Saving model ...
    Epoch: 57 	Training Loss: 3.811537 	Validation Loss: 3.795472	Elapsed: 0:01:15.602536
    Epoch: 58 	Training Loss: 3.805120 	Validation Loss: 3.781486	Elapsed: 0:01:15.072221
    Epoch: 59 	Training Loss: 3.774735 	Validation Loss: 3.759193	Elapsed: 0:01:14.986689
    Validation loss decreased (3.777117 --> 3.759193). Saving model ...
    Epoch: 60 	Training Loss: 3.775809 	Validation Loss: 3.760698	Elapsed: 0:01:15.881241
    Epoch: 61 	Training Loss: 3.747226 	Validation Loss: 3.727979	Elapsed: 0:01:15.143611
    Validation loss decreased (3.759193 --> 3.727979). Saving model ...
    Epoch: 62 	Training Loss: 3.740218 	Validation Loss: 3.651795	Elapsed: 0:01:16.844629
    Validation loss decreased (3.727979 --> 3.651795). Saving model ...
    Epoch: 63 	Training Loss: 3.705042 	Validation Loss: 3.679952	Elapsed: 0:01:16.842594
    Epoch: 64 	Training Loss: 3.708105 	Validation Loss: 3.672383	Elapsed: 0:01:16.632940
    Epoch: 65 	Training Loss: 3.665279 	Validation Loss: 3.666996	Elapsed: 0:01:17.039767
    Epoch: 66 	Training Loss: 3.678146 	Validation Loss: 3.622979	Elapsed: 0:01:16.949481
    Validation loss decreased (3.651795 --> 3.622979). Saving model ...
    Epoch: 67 	Training Loss: 3.648702 	Validation Loss: 3.737132	Elapsed: 0:01:16.997128
    Epoch: 68 	Training Loss: 3.651722 	Validation Loss: 3.628284	Elapsed: 0:01:15.263572
    Epoch: 69 	Training Loss: 3.641331 	Validation Loss: 3.610956	Elapsed: 0:01:15.177596
    Validation loss decreased (3.622979 --> 3.610956). Saving model ...
    Epoch: 70 	Training Loss: 3.606314 	Validation Loss: 3.603912	Elapsed: 0:01:14.972808
    Validation loss decreased (3.610956 --> 3.603912). Saving model ...
    Epoch: 71 	Training Loss: 3.604512 	Validation Loss: 3.563650	Elapsed: 0:01:14.961408
    Validation loss decreased (3.603912 --> 3.563650). Saving model ...
    Epoch: 72 	Training Loss: 3.577432 	Validation Loss: 3.591363	Elapsed: 0:01:15.020969
    Epoch: 73 	Training Loss: 3.532477 	Validation Loss: 3.581723	Elapsed: 0:01:15.028302
    Epoch: 74 	Training Loss: 3.557664 	Validation Loss: 3.609078	Elapsed: 0:01:14.897522
    Epoch: 75 	Training Loss: 3.534345 	Validation Loss: 3.660126	Elapsed: 0:01:15.057398
    Epoch: 76 	Training Loss: 3.528575 	Validation Loss: 3.554802	Elapsed: 0:01:14.930904
    Validation loss decreased (3.563650 --> 3.554802). Saving model ...
    Epoch: 77 	Training Loss: 3.533098 	Validation Loss: 3.548549	Elapsed: 0:01:14.751885
    Validation loss decreased (3.554802 --> 3.548549). Saving model ...
    Epoch: 78 	Training Loss: 3.510743 	Validation Loss: 3.542639	Elapsed: 0:01:15.279958
    Validation loss decreased (3.548549 --> 3.542639). Saving model ...
    Epoch: 79 	Training Loss: 3.481106 	Validation Loss: 3.511475	Elapsed: 0:01:14.732100
    Validation loss decreased (3.542639 --> 3.511475). Saving model ...
    Epoch: 80 	Training Loss: 3.457741 	Validation Loss: 3.544109	Elapsed: 0:01:14.576692
    Epoch: 81 	Training Loss: 3.477495 	Validation Loss: 3.502242	Elapsed: 0:01:14.692703
    Validation loss decreased (3.511475 --> 3.502242). Saving model ...
    Epoch: 82 	Training Loss: 3.438451 	Validation Loss: 3.494920	Elapsed: 0:01:14.448274
    Validation loss decreased (3.502242 --> 3.494920). Saving model ...
    Epoch: 83 	Training Loss: 3.404955 	Validation Loss: 3.472970	Elapsed: 0:01:14.875598
    Validation loss decreased (3.494920 --> 3.472970). Saving model ...
    Epoch: 84 	Training Loss: 3.398070 	Validation Loss: 3.458972	Elapsed: 0:01:14.739727
    Validation loss decreased (3.472970 --> 3.458972). Saving model ...
    Epoch: 85 	Training Loss: 3.416896 	Validation Loss: 3.472682	Elapsed: 0:01:15.312842
    Epoch: 86 	Training Loss: 3.378686 	Validation Loss: 3.468552	Elapsed: 0:01:14.883741
    Epoch: 87 	Training Loss: 3.368700 	Validation Loss: 3.444025	Elapsed: 0:01:14.862004
    Validation loss decreased (3.458972 --> 3.444025). Saving model ...
    Epoch: 88 	Training Loss: 3.367541 	Validation Loss: 3.435218	Elapsed: 0:01:15.151076
    Validation loss decreased (3.444025 --> 3.435218). Saving model ...
    Epoch: 89 	Training Loss: 3.329758 	Validation Loss: 3.446027	Elapsed: 0:01:15.140366
    Epoch: 90 	Training Loss: 3.341512 	Validation Loss: 3.420673	Elapsed: 0:01:14.745760
    Validation loss decreased (3.435218 --> 3.420673). Saving model ...
    Epoch: 91 	Training Loss: 3.322894 	Validation Loss: 3.420635	Elapsed: 0:01:15.167178
    Validation loss decreased (3.420673 --> 3.420635). Saving model ...
    Epoch: 92 	Training Loss: 3.324355 	Validation Loss: 3.412082	Elapsed: 0:01:15.110058
    Validation loss decreased (3.420635 --> 3.412082). Saving model ...
    Epoch: 93 	Training Loss: 3.294182 	Validation Loss: 3.421644	Elapsed: 0:01:14.638429
    Epoch: 94 	Training Loss: 3.280874 	Validation Loss: 3.445339	Elapsed: 0:01:15.124445
    Epoch: 95 	Training Loss: 3.287893 	Validation Loss: 3.407052	Elapsed: 0:01:15.223214
    Validation loss decreased (3.412082 --> 3.407052). Saving model ...
    Epoch: 96 	Training Loss: 3.279562 	Validation Loss: 3.382630	Elapsed: 0:01:15.119832
    Validation loss decreased (3.407052 --> 3.382630). Saving model ...
    Epoch: 97 	Training Loss: 3.251223 	Validation Loss: 3.354045	Elapsed: 0:01:15.125025
    Validation loss decreased (3.382630 --> 3.354045). Saving model ...
    Epoch: 98 	Training Loss: 3.234788 	Validation Loss: 3.364112	Elapsed: 0:01:15.093847
    Epoch: 99 	Training Loss: 3.223984 	Validation Loss: 3.362896	Elapsed: 0:01:15.448628
    Epoch: 100 	Training Loss: 3.188159 	Validation Loss: 3.369453	Elapsed: 0:01:15.560517
    Training Ended: 2019-04-01 09:51:25.808831
    Total Training Time: 2:05:27.025177


### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images.  Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 10%.


```python
model_scratch.load_state_dict(torch.load(scratch_path))
```


```python
def test(loaders, model, criterion, use_cuda, print_function=print):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print_function('Test Loss: {:.6f}\n'.format(test_loss))

    print_function('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
```


```python
scratch_test_log = Tee("scratch_test.log", directory_name=MODEL_PATH)
```


```python
# call test function    
test(loaders_scratch, model_scratch, criterion_scratch, use_cuda, print_function=scratch_test_log)
```

    Test Loss: 3.369061
    
    
    Test Accuracy: 18% (157/836)


---
<a id='step4'></a>
## Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

### (IMPLEMENTATION) Specify Data Loaders for the Dog Dataset

Use the code cell below to write three separate [data loaders](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) for the training, validation, and test datasets of dog images (located at `dogImages/train`, `dogImages/valid`, and `dogImages/test`, respectively). 

If you like, **you are welcome to use the same data loaders from the previous step**, when you created a CNN from scratch.


```python
## TODO: Specify data loaders
loaders_transfer = loaders_scratch
```

### (IMPLEMENTATION) Model Architecture

Use transfer learning to create a CNN to classify dog breed.  Use the code cell below, and save your initialized model as the variable `model_transfer`.


```python
import torchvision.models as models
import torch.nn as nn

## TODO: Specify model architecture 

model_transfer = models.vgg16(pretrained=True)

for param in model_transfer.features.parameters():
    param.require_grad = False

num_features = model_transfer.classifier[6].in_features
features = list(model_transfer.classifier.children())[:-1] 
features.extend([nn.Linear(num_features, BREEDS)]) 
model_transfer.classifier = nn.Sequential(*features)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

if use_cuda:
    model_transfer = model_transfer.cuda()
```


```python
device = torch.device("cuda" if use_cuda else "cpu")
print("Using {}".format(device))
```

    Using cuda



```python
print(model_transfer.classifier[6].out_features)
```

    133



```python
model_transfer
```




    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace)
        (2): Dropout(p=0.5)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace)
        (5): Dropout(p=0.5)
        (6): Linear(in_features=4096, out_features=133, bias=True)
      )
    )



__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__  The VGG network is characterized by its simplicity, using only 3×3 convolutional layers stacked on top of each other in increasing depth. Reducing volume size is handled by max pooling. Two fully-connected layers, each with 4,096 nodes are then followed by a softmax classifier. 

I used the classification using VGG16, which has originally 1000 outputs. I changed final layer to connect 133 outputs. VGG16 is one of the well-perform classifier and pretrained model. Additional to that, this model can be implemented in my scratch code `train` and `test`. 
VGG16 is well structured and I checked the final layer is consist of input 4096 and output 1000. I changed the final output to 133, because our target class number (BREEDS) is 133. 

### (IMPLEMENTATION) Specify Loss Function and Optimizer

Use the next code cell to specify a [loss function](http://pytorch.org/docs/master/nn.html#loss-functions) and [optimizer](http://pytorch.org/docs/master/optim.html).  Save the chosen loss function as `criterion_transfer`, and the optimizer as `optimizer_transfer` below.


```python
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optimizer.SGD(
    model_transfer.parameters(),
    lr=0.001,
    momentum=0.9)
```

### (IMPLEMENTATION) Train and Validate the Model

Train and validate your model in the code cell below.  [Save the final model parameters](http://pytorch.org/docs/master/notes/serialization.html) at filepath `'model_transfer.pt'`.


```python
transfer_model_path = MODEL_PATH.joinpath("model_transfer.pt")
```


```python
transfer_log = Tee(log_name="transfer_train.log", directory_name=MODEL_PATH)
```


```python
EPOCHS = 10
```


```python
# train the model

model_transfer = train(EPOCHS,
                       loaders=loaders_transfer,
                       model=model_transfer,
                       optimizer=optimizer_transfer,
                       criterion=criterion_transfer,
                       use_cuda=use_cuda,
                       save_path=transfer_model_path,
                       print_function=transfer_log)
```

    Training Started: 2019-04-01 09:51:34.781347
    Epoch: 1 	Training Loss: 2.642515 	Validation Loss: 1.021088	Elapsed: 0:02:01.426559
    Validation loss decreased (inf --> 1.021088). Saving model ...
    Epoch: 2 	Training Loss: 1.461885 	Validation Loss: 0.918598	Elapsed: 0:02:03.815928
    Validation loss decreased (1.021088 --> 0.918598). Saving model ...
    Epoch: 3 	Training Loss: 1.293269 	Validation Loss: 0.789486	Elapsed: 0:02:03.573854
    Validation loss decreased (0.918598 --> 0.789486). Saving model ...
    Epoch: 4 	Training Loss: 1.169967 	Validation Loss: 0.735015	Elapsed: 0:02:03.663114
    Validation loss decreased (0.789486 --> 0.735015). Saving model ...
    Epoch: 5 	Training Loss: 1.122108 	Validation Loss: 0.726724	Elapsed: 0:02:03.789837
    Validation loss decreased (0.735015 --> 0.726724). Saving model ...
    Epoch: 6 	Training Loss: 1.073918 	Validation Loss: 0.747401	Elapsed: 0:02:03.513132
    Epoch: 7 	Training Loss: 1.007255 	Validation Loss: 0.606368	Elapsed: 0:02:04.093370
    Validation loss decreased (0.726724 --> 0.606368). Saving model ...
    Epoch: 8 	Training Loss: 1.003418 	Validation Loss: 0.647689	Elapsed: 0:02:03.597189
    Epoch: 9 	Training Loss: 0.944159 	Validation Loss: 0.754289	Elapsed: 0:02:03.919422
    Epoch: 10 	Training Loss: 0.894426 	Validation Loss: 0.641945	Elapsed: 0:02:03.901985
    Training Ended: 2019-04-01 10:12:25.328107
    Total Training Time: 0:20:50.546760


### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Use the code cell below to calculate and print the test loss and accuracy.  Ensure that your test accuracy is greater than 60%.


```python
transfer_test_log = Tee(log_name="transfer_test.log", directory_name=MODEL_PATH)
```


```python
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda, print_function=transfer_test_log)
```

    Test Loss: 0.702561
    
    
    Test Accuracy: 80% (671/836)


### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan hound`, etc) that is predicted by your model.  


```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in training.classes]

def predict_breed_transfer(image_path):
    # load the image
    image = Image.open(image_path)

    # convert the image to a tensor
    tensor = test_transform(image)

    # add a batch number
    tensor = tensor.unsqueeze_(0)

    # put on the GPU or CPU
    tensor = tensor.to(device)

    # make it a variable
    x = torch.autograd.Variable(tensor)

    # make the prediction
    output = model_transfer(x)
    return class_names[output.data.cpu().numpy().argmax()]
```

---
<a id='step5'></a>
## Step 5: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `human_detector` functions developed above.  You are __required__ to use your CNN from Step 4 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


### (IMPLEMENTATION) Write your Algorithm


```python
import os 
import numpy as np
from functools import partial
from tabulate import tabulate
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("{}/data/lfw/*/*".format(os.getcwd())))
dog_files = np.array(glob("{}/data/dog_images/*/*/*".format(os.getcwd())))
test_files = np.array(glob("{}/data/test/*".format(os.getcwd())))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
print('There are %d total test images.' % len(test_files))
```

    There are 13233 total human images.
    There are 8351 total dog images.
    There are 6 total test images.



```python
from tqdm import tqdm

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]
```


```python
import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()
```


```python
device = torch.device("cuda" if use_cuda else "cpu")
print("Using {}".format(device))
```

    Using cuda


Human detector


```python
import cv2                

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

def human_detector(img_path):
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

Dog detector


```python
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

IMAGE_SIZE = 224
IMAGE_HALF_SIZE = IMAGE_SIZE//2

vgg_transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
```


```python
def model_predict(image_path: str, model: nn.Module, transform: transforms.Compose):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    
    image = Image.open(str(image_path))
    image = transform(image).unsqueeze(0).to(device)
    output = model(image)
    probabilities = torch.exp(output)
    _, top_class = probabilities.topk(1, dim=1)
    return top_class.item()   

VGG16_predict = partial(model_predict, model=VGG16, transform=vgg_transform)
```


```python
DOG_LOWER, DOG_UPPER = 150, 268
```


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path, predictor: callable=VGG16_predict):
    ## TODO: Complete the function.
    
    return DOG_LOWER < VGG16_predict(img_path) < DOG_UPPER
```


```python
import matplotlib.pyplot as plt
pyplot = plt

def render(image_path: str, species: str, breed: str):
    """Renders the image

    Args:
     image_path: path to the image to render
     species: identified species
     breed: identified breed
    """
    name = " ".join(image_path.name.split(".")[0].split("_")).title()
    figure, axe = pyplot.subplots()
    figure.suptitle("{} ({})".format(species, name), weight="bold")
    axe.set_xlabel("Looks like a {}.".format(breed))
    image = Image.open(image_path)
    axe.tick_params(dict(axis="both",
                         which="both",
                         bottom=False,
                         top=False))
    axe.get_xaxis().set_ticks([])
    axe.get_yaxis().set_ticks([])
    axe_image = axe.imshow(image)
    return
```


```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def run_app(img):
    ## handle cases for a human face, dog, and neither
    image_path = Path(img)
    is_dog = dog_detector(image_path)
    is_human = human_detector(image_path)

    if not is_dog and not is_human:
        species = "Error: Neither Human nor Dog"
        breed = "?"
    else:
        breed = predict_breed_transfer(image_path)

    if is_dog and is_human:
        species = "Human-Dog Hybrid"
    elif is_dog:
        species = "Dog"
    elif is_human:
        species = "Human"
    render(image_path, species, breed)
    return
```

---
<a id='step6'></a>
## Step 6: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that _you_ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ (Three possible points for improvement)
* More dataset
* Hyper-parameter tuning process
* Model ensemble
* Additional techniques (ex. Meta-learning)


```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

## suggested code, below
for file in np.hstack((human_files[:3], dog_files[:3])):
    run_app(file)
```


![png](output_126_0.png)



![png](output_126_1.png)



![png](output_126_2.png)



![png](output_126_3.png)



![png](output_126_4.png)



![png](output_126_5.png)



```python
test_files = sorted(test_files)

for file in np.hstack((test_files)):
    run_app(file)
```


![png](output_127_0.png)



![png](output_127_1.png)



![png](output_127_2.png)



![png](output_127_3.png)



![png](output_127_4.png)



![png](output_127_5.png)

