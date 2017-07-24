#!/usr/bin/env python


"""
Testing all the nnet library
"""
from __future__ import division, print_function

import numpy
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score

from hep_ml import nnet
from hep_ml.commonutils import generate_sample
from hep_ml.preprocessing import BinTransformer, IronTransformer

__author__ = 'Alex Rogozhnikov'


def test_nnet(n_samples=200, n_features=7, distance=0.8, complete=False):
    """
    :param complete: if True, all possible combinations will be checked, and quality is printed
    """
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)

    nn_types = [
        nnet.SimpleNeuralNetwork,
        nnet.MLPClassifier,
        nnet.SoftmaxNeuralNetwork,
        nnet.RBFNeuralNetwork,
        nnet.PairwiseNeuralNetwork,
        nnet.PairwiseSoftplusNeuralNetwork,
    ]

    if complete:
        # checking all possible combinations
        for loss in nnet.losses:
            for NNType in nn_types:
                for trainer in nnet.trainers:
                    nn = NNType(layers=[5], loss=loss, trainer=trainer, random_state=42)
                    nn.fit(X, y, epochs=100)
                    print(roc_auc_score(y, nn.predict_proba(X)[:, 1]), nn)

        lr = LogisticRegression().fit(X, y)
        print(lr, roc_auc_score(y, lr.predict_proba(X)[:, 1]))

        assert 0 == 1, "Let's see and compare results"
    else:
        # checking combinations of losses, nn_types, trainers, most of them are used once during tests.
        attempts = max(len(nnet.losses), len(nnet.trainers), len(nn_types))
        losses_shift = numpy.random.randint(10)
        trainers_shift = numpy.random.randint(10)
        for attempt in range(attempts):
            # each combination is tried 3 times. before raising exception
            retry_attempts = 3
            for retry_attempt in range(retry_attempts):
                loss = list(nnet.losses.keys())[(attempt + losses_shift) % len(nnet.losses)]
                trainer = list(nnet.trainers.keys())[(attempt + trainers_shift) % len(nnet.trainers)]

                nn_type = nn_types[attempt % len(nn_types)]

                nn = nn_type(layers=[5], loss=loss, trainer=trainer, random_state=42 + retry_attempt)
                print(nn)
                nn.fit(X, y, epochs=200)
                quality = roc_auc_score(y, nn.predict_proba(X)[:, 1])
                computed_loss = nn.compute_loss(X, y)
                if quality > 0.8:
                    break
                else:
                    print('attempt {} : {}'.format(retry_attempt, quality))
                    if retry_attempt == retry_attempts - 1:
                        raise RuntimeError('quality of model is too low: {} {}'.format(quality, nn))


def test_with_scaler(n_samples=200, n_features=15, distance=0.5):
    X, y = generate_sample(n_samples=n_samples, n_features=n_features, distance=distance)
    for scaler in [BinTransformer(max_bins=16), IronTransformer()]:
        clf = nnet.SimpleNeuralNetwork(scaler=scaler)
        clf.fit(X, y, epochs=300)

        p = clf.predict_proba(X)
        assert roc_auc_score(y, p[:, 1]) > 0.8, 'quality is too low for model: {}'.format(clf)


print("NNet test")
test_nnet()
print("Scaler test")
test_with_scaler()
