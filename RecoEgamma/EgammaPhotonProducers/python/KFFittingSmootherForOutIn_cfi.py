import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.KFTrajectoryFitterForOutIn_cfi import *
from RecoEgamma.EgammaPhotonProducers.KFTrajectorySmootherForOutIn_cfi import *
import TrackingTools.TrackFitters.KFFittingSmoother_cfi
# KFFittingSmootherESProducer
KFFittingSmootherForOutIn = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone(
    ComponentName   = 'KFFittingSmootherForOutIn',
    Fitter          = 'KFFitterForOutIn',
    Smoother        = 'KFSmootherForOutIn',
    EstimateCut     = -1,
    MinNumberOfHits = 3
)
