import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.KFTrajectoryFitterForInOut_cfi import *
from RecoEgamma.EgammaPhotonProducers.KFTrajectorySmootherForInOut_cfi import *
import TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi
# KFFittingSmootherESProducer
KFFittingSmootherForInOut = TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi.KFFittingSmoother.clone()
KFFittingSmootherForInOut.ComponentName = 'KFFittingSmootherForInOut'
KFFittingSmootherForInOut.Fitter = 'KFFitterForInOut'
KFFittingSmootherForInOut.Smoother = 'KFSmootherForInOut'
KFFittingSmootherForInOut.EstimateCut = -1
KFFittingSmootherForInOut.MinNumberOfHits = 3

