import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.propAlongMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.chi2EstimatorForInOutFit_cfi import *
import TrackingTools.TrackFitters.KFTrajectoryFitter_cfi
#KFTrajectoryFitterESProducer
KFTrajectoryFitterForInOut = TrackingTools.TrackFitters.KFTrajectoryFitter_cfi.KFTrajectoryFitter.clone()
KFTrajectoryFitterForInOut.ComponentName = 'KFFitterForInOut'
KFTrajectoryFitterForInOut.Propagator = 'alongMomElePropagator'
KFTrajectoryFitterForInOut.Estimator = 'Chi2ForInOut'
KFTrajectoryFitterForInOut.minHits = 3

