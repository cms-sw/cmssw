import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.KFTrajectoryFitterForOutIn_cfi import *
from RecoEgamma.EgammaPhotonProducers.KFTrajectorySmootherForOutIn_cfi import *
import copy
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# KFFittingSmootherESProducer
KFFittingSmootherForOutIn = copy.deepcopy(KFFittingSmoother)
KFFittingSmootherForOutIn.ComponentName = 'KFFittingSmootherForOutIn'
KFFittingSmootherForOutIn.Fitter = 'KFFitterForOutIn'
KFFittingSmootherForOutIn.Smoother = 'KFSmootherForOutIn'
KFFittingSmootherForOutIn.EstimateCut = -1
KFFittingSmootherForOutIn.MinNumberOfHits = 3

