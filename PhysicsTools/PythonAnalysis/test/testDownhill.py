#!/usr/bin/env python

import downhill
import numpy as np
import theano
import theano.tensor as TT

FLOAT = 'df'[theano.config.floatX == 'float32']

def rand(a, b):
    return np.random.randn(a, b).astype(FLOAT)

A, B, K = 20, 5, 3

# Set up a matrix factorization problem to optimize.
u = theano.shared(rand(A, K), name='u')
v = theano.shared(rand(K, B), name='v')
z = TT.matrix()
err = TT.sqr(z - TT.dot(u, v))
loss = err.mean() + abs(u).mean() + (v * v).mean()

# Minimize the regularized loss with respect to a data matrix.
y = np.dot(rand(A, K), rand(K, B)) + rand(A, B)

# Monitor during optimization.
monitors = (('err', err.mean()),
            ('|u|<0.1', (abs(u) < 0.1).mean()),
            ('|v|<0.1', (abs(v) < 0.1).mean()))

#minimize(
#loss, 
#train, 
#batch_size=32, 
#monitor_gradients=False, 
#monitors=(), 

#valid=None, 
#params=None, 
#inputs=None, 
#algo='rmsprop', 
#updates=(), 
#train_batches=None, 
#valid_batches=None, 
#**kwargs)

downhill.minimize(
    loss=loss,
    train=[y],
    patience=0,
    batch_size=A,                 # Process y as a single batch.
    max_gradient_norm=1,          # Prevent gradient explosion!
    learning_rate=0.1,
    monitors=monitors,
    monitor_gradients=True)

# Print out the optimized coefficients u and basis v.
print('u =', u.get_value())
print('v =', v.get_value())
