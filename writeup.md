
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/samples.png "Samples"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/augmented.png "Augmented"
[image4]: ./examples/test5.png "Test5"
[image5]: ./examples/result5.png "Result5"
[image6]: ./examples/softmax1.png "Softmax1"
[image7]: ./examples/softmax2.png "Softmax2"
[image8]: ./examples/softmax3.png "Softmax3"
[image9]: ./examples/softmax4.png "Softmax4"
[image10]: ./examples/softmax5.png "Softmax5"
[image11]: ./examples/visualization.png "Visualization"
[image12]: ./examples/original.png "Original"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/wang-yang/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34,799`
* The size of the validation set is `4,410`
* The size of test set is `12,630`
* The shape of a traffic sign image is `32px X 32px in RGB`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

5 randomly selected image from data set:

![Samples][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because shape is more important than color for traffic signs. And also can convert the RGB colour 3 channel space into a single grayscale space.

Here is an example of a traffic sign image before and after grayscaling.

As a last step, I normalized the image data so that the data is distributed uniformly around zero because it will speed up the convergence procedure and improve the quality of model.

The result of grayscaling and normalization is:

![Grayscale][image2]

The code cell is from `#7` to `#11` in `./Traffic_Sign_Classifier.ipynb`

Then I decided to generate additional data because after running a couple of training experiments, I found the model tends to be overfitted. The prediction accuracy is around 89%. The model rely too much on the traffic sign position. We need to generate some augmented image which have more variety position.

To add more data to the the data set, I generate 5 additional image for each original image.
Each newly generated image is randomly shifting or rotating from the original image.

The code cell is `#12` in `./Traffic_Sign_Classifier.ipynb` as bellow:

```python
from scipy.ndimage.interpolation import shift, rotate

def gen_jittered(image, repeat):
    for i in range(0, repeat):
        yield jitter(np.copy(image))   

def jitter(img):
    sx = random.randint(-2, 2)
    sy = random.randint(-2, 2)
    d = random.randint(-15, 15)
    return rotate(shift(img, [sy, sx, 0]), d, mode = "nearest", reshape = False)

GEN_NUM = 5
shape = X_train_norm.shape
generate_total = shape[0] * GEN_NUM
new_X = np.empty((generate_total, shape[1], shape[2], shape[3]), dtype = X_train_norm.dtype)
new_Y = np.empty((generate_total,), dtype = y_train.dtype)
for i in range(0, len(X_train_norm)):
    offset = i * GEN_NUM
    end = offset + GEN_NUM
    new_X[offset:end] = np.array(list(gen_jittered(X_train_norm[i], GEN_NUM)))
    new_Y[offset:end] = np.full(GEN_NUM, y_train[i])
```

With the `jitter()` function, shifting was performed in both dimenstions by [-2px, +2px].
And the image wa rotate by[-15 degree, +15 degree].

In total, there are `173,995` augmented images were added into Data set.

Here is an example of an original image and an augmented image:

![Augmented][image3]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I use LeNet as my neural network. And it consisted of the following layers:

| Layer                 |     Description                               |
|:--------------------- | ---------------------------------------------:|
| Input                 | 32x32x1 Grayscale image                       |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x10   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 14x14x10   |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  | -                                             |
| Max pooling           | 2x2 stride, valid padding, outputs 5x5x16     |
| Fully connected       | Input 400, output 120                         |
| RELU                  | -                                             |
| Fully connected       | Input 120, output 84                          |
| RELU                  | -                                             |
| Fully connected       | Input 84, output 43                           |
| Softmax               | -                                             |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam optimizer and cross-entropy as a loss function, `cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)`

Other hyper-parameters are:
* number of epochs = 20
* batch size = 256
* learning rate = 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of `0.99`
* validation set accuracy of `0.936`
* test set accuracy of `0.936`

In order to achieve at least 0.93 accuracy on the validation set, I decided to use the LeNet architecture. Because the LeNet was designed for MNIST prediction, and these handwritten characters are very similar to grayscaled traffic sign images. So my assumption is that the LeNet model can generate same level of accuracy for traffic sign prediction as MNIST.

The initial experiments on the original data set only produced a 89% accuracy on the validation set. However, after the apply the image augmentation, the accuracy increased to 0.936.

###Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Test5][image4]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                 | Prediction                        |
|:--------------------- | ---------------------------------:|
| General caution       | General caution                   |
| Slippery road         | Slippery road                     |
| Speed limit (70km/h)  | Speed limit (70km/h)              |
| Children crossing     | Right of way at next intersection |
| No entry              | No entry                          |

![Result5][image5]

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 0.93

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the `#25` cell of the Ipython notebook.

For the first image, the model prediction was correct : "General caution" sign with with probability as bellow:

![Softmax distribution for the image 1][image6]

For the second image, the model prediction was correct : "Slippery road" sign with probability as bellow:

![Softmax distribution for the image 2][image7]

For the third image, the model prediction was correct : "Speed limit (70km/h)" sign with probability as bellow:

![Softmax distribution for the image 3][image8]

For the fourth image, the model prediction was worng : "Road narrows on the right" sign, but the true label is "Children crossing", the probability distribution is as bellow:

![Softmax distribution for the image 4][image9]

For the fifth image, the model prediction was correct : "No entry" sign with probability as bellow:

![Softmax distribution for the image 5][image10]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Let's see the original image :

![Original][image12]

And the visualization of each feature map at the first layer of LeNet:

![Visualization][image11]

Visualization is a good way to help me to understanding how deep convolutional neural network capture the feature of image. Each feature map is activated by different part of edges of image or shapes of image. 
