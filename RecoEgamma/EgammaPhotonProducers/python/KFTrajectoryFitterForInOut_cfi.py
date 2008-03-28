import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.propAlongMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.chi2EstimatorForInOutFit_cfi import *
import copy
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
#KFTrajectoryFitterESProducer
KFTrajectoryFitterForInOut = copy.deepcopy(KFTrajectoryFitter)
KFTrajectoryFitterForInOut.ComponentName = 'KFFitterForInOut'
KFTrajectoryFitterForInOut.Propagator = 'alongMomElePropagator'
KFTrajectoryFitterForInOut.Estimator = 'Chi2ForInOut'
KFTrajectoryFitterForInOut.minHits = 3

