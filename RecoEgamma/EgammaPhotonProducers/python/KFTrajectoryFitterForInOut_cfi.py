import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.propAlongMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.chi2EstimatorForInOutFit_cfi import *
import TrackingTools.TrackFitters.KFTrajectoryFitter_cfi
#KFTrajectoryFitterESProducer
KFTrajectoryFitterForInOut = TrackingTools.TrackFitters.KFTrajectoryFitter_cfi.KFTrajectoryFitter.clone(
    ComponentName = 'KFFitterForInOut',
    Propagator    = 'alongMomElePropagator',
    Estimator     = 'Chi2ForInOut',
    minHits       = 3
)
