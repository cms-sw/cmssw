import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.propAlongMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.chi2EstimatorForOutInFit_cfi import *
import copy
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
#KFTrajectoryFitterESProducer
KFTrajectoryFitterForOutIn = copy.deepcopy(KFTrajectoryFitter)
KFTrajectoryFitterForOutIn.ComponentName = 'KFFitterForOutIn'
KFTrajectoryFitterForOutIn.Propagator = 'alongMomElePropagator'
KFTrajectoryFitterForOutIn.Estimator = 'Chi2ForOutIn'
KFTrajectoryFitterForOutIn.minHits = 3

