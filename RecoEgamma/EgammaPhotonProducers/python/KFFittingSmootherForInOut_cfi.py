import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.KFTrajectoryFitterForInOut_cfi import *
from RecoEgamma.EgammaPhotonProducers.KFTrajectorySmootherForInOut_cfi import *
import TrackingTools.TrackFitters.KFFittingSmoother_cfi
# KFFittingSmootherESProducer
KFFittingSmootherForInOut = TrackingTools.TrackFitters.KFFittingSmoother_cfi.KFFittingSmoother.clone(
    ComponentName   = 'KFFittingSmootherForInOut',
    Fitter          = 'KFFitterForInOut',
    Smoother        = 'KFSmootherForInOut',
    EstimateCut     = -1,
    MinNumberOfHits = 3
)
