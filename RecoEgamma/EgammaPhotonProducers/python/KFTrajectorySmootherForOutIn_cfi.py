import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.propOppoMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.chi2EstimatorForOutInFit_cfi import *
import TrackingTools.TrackFitters.KFTrajectorySmoother_cfi
# KFTrajectorySmootherESProducer
KFTrajectorySmootherForOutIn = TrackingTools.TrackFitters.KFTrajectorySmoother_cfi.KFTrajectorySmoother.clone(
    ComponentName = 'KFSmootherForOutIn',
    Propagator    = 'oppositeToMomElePropagator',
    Estimator     = 'Chi2ForOutIn'
)
