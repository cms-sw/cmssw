import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.propOppoMomentumWithMaterialForElectrons_cfi import *
from RecoEgamma.EgammaPhotonProducers.chi2EstimatorForInOutFit_cfi import *
import copy
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFTrajectorySmootherESProducer
KFTrajectorySmootherForInOut = copy.deepcopy(KFTrajectorySmoother)
KFTrajectorySmootherForInOut.ComponentName = 'KFSmootherForInOut'
KFTrajectorySmootherForInOut.Propagator = 'oppositeToMomElePropagator'
KFTrajectorySmootherForInOut.Estimator = 'Chi2ForInOut'

