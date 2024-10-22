import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.propAlongMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.chi2EstimatorForOutInFit_cfi import *
import TrackingTools.TrackFitters.KFTrajectoryFitter_cfi
#KFTrajectoryFitterESProducer
KFTrajectoryFitterForOutIn = TrackingTools.TrackFitters.KFTrajectoryFitter_cfi.KFTrajectoryFitter.clone(
    ComponentName = 'KFFitterForOutIn',
    Propagator    = 'alongMomElePropagator',
    Estimator     = 'Chi2ForOutIn',
    minHits       = 3
)
