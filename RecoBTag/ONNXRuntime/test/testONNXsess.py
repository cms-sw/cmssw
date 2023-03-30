import onnxruntime as ort
import numpy as np
np.random.seed = 42
print('ONNXRuntime version: ', ort.__version__)
# create input data in the float format (32 bit)

data1 = np.ones((1, 25, 16)).astype(np.float32)
#data2 = np.ones((1, 25,  8)).astype(np.float32)
data2 = np.random.rand(1, 25,  8).astype(np.float32)
#data3 = np.ones((1,  5, 14)).astype(np.float32)
data3 = np.array([[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
  [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
 # [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
  [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0]
                  ]
                 ]).astype(np.float32)
data4 = np.ones((1, 25,  4)).astype(np.float32)
data5 = np.ones((1, 25,  4)).astype(np.float32)
#data6 = np.zeros((1,  5,  4)).astype(np.float32)
data6 = np.array([[[1., 1., 1., 1.],
  [1., 1., 1., 1.],
  [1., 1., 1., 1.],
  [1., 1., 1., 1.],
#  [1., 1., 1., 1.],
  [0., 0., 0., 0.]
                  ]
                 ]).astype(np.float32)
# create inference session using ort.InferenceSession from a given model
ort_sess = ort.InferenceSession('/afs/cern.ch/work/a/anstein/private/ParTPUPPI/CMSSW_13_1_0_pre2/src/RecoBTag/Combined/data/RobustParTAK4/PUPPI/V00/DJT.onnx', providers=[
    #'CUDAExecutionProvider', 
                                                                 'CPUExecutionProvider'])

# run inference
outputs = ort_sess.run(None, {'input_1': data1,
                              'input_2': data2,
                              'input_3': data3,
                              'input_4': data4,
                              'input_5': data5,
                              'input_6': data6})[0]

# print input and output
# print('input ->', data)
#print('input ->', data1, data2, data3, data4, data5, data6)
print('output ->', outputs)



#  data1 = np.ones((1, 25, 16)).astype(np.float32)
#  data2 = np.ones((1, 25,  8)).astype(np.float32)
#  data3 = np.ones((1,  4, 14)).astype(np.float32)
#  data4 = np.ones((1, 25,  4)).astype(np.float32)
#  data5 = np.ones((1, 25,  4)).astype(np.float32)
#  data6 = np.ones((1,  4,  4)).astype(np.float32)
# create inference session using ort.InferenceSessio
# create inference session using ort.InferenceSession from a given model
ort_sess = ort.InferenceSession('/afs/cern.ch/work/a/anstein/private/ParTPUPPI/CMSSW_13_1_0_pre2/src/RecoBTag/Combined/data/RobustParTAK4/PUPPI/V00/DJT_build4v.onnx', providers=[
    #'CUDAExecutionProvider', 
                                                                 'CPUExecutionProvider'])

# run inference
outputs = ort_sess.run(None, {'input_1': data1,
                              'input_2': data2,
                              'input_3': data3,
                              'input_4': data4,
                              'input_5': data5,
                              'input_6': data6})[0]

# print input and output
# print('input ->', data)
#  print('input ->', data1, data2, data3, data4, data5, data6)
print('output ->', outputs)