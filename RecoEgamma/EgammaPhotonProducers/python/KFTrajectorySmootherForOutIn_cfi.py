import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.propOppoMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.chi2EstimatorForOutInFit_cfi import *
import copy
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFTrajectorySmootherESProducer
KFTrajectorySmootherForOutIn = copy.deepcopy(KFTrajectorySmoother)
KFTrajectorySmootherForOutIn.ComponentName = 'KFSmootherForOutIn'
KFTrajectorySmootherForOutIn.Propagator = 'oppositeToMomElePropagator'
KFTrajectorySmootherForOutIn.Estimator = 'Chi2ForOutIn'

