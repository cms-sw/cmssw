import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.KFTrajectoryFitterForInOut_cfi import *
from RecoEgamma.EgammaPhotonProducers.KFTrajectorySmootherForInOut_cfi import *
import copy
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# KFFittingSmootherESProducer
KFFittingSmootherForInOut = copy.deepcopy(KFFittingSmoother)
KFFittingSmootherForInOut.ComponentName = 'KFFittingSmootherForInOut'
KFFittingSmootherForInOut.Fitter = 'KFFitterForInOut'
KFFittingSmootherForInOut.Smoother = 'KFSmootherForInOut'
KFFittingSmootherForInOut.EstimateCut = -1
KFFittingSmootherForInOut.MinNumberOfHits = 3

