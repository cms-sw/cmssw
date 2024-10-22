import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.propOppoMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.chi2EstimatorForInOutFit_cfi import *
import TrackingTools.TrackFitters.KFTrajectorySmoother_cfi
# KFTrajectorySmootherESProducer
KFTrajectorySmootherForInOut = TrackingTools.TrackFitters.KFTrajectorySmoother_cfi.KFTrajectorySmoother.clone(
    ComponentName = 'KFSmootherForInOut',
    Propagator    = 'oppositeToMomElePropagator',
    Estimator     = 'Chi2ForInOut'
)
