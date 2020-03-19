#!/usr/bin/env python
import mxnet as mx
import numpy
a = mx.nd.ones((2, 3))
b = a * 2 + 1
b.asnumpy()
numpy.array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=numpy.float32)
