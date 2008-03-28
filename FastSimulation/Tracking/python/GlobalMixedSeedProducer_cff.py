import FWCore.ParameterSet.Config as cms

import copy
from FastSimulation.Tracking.TrajectorySeedProducer_cfi import *
globalMixedGSSeeds = copy.deepcopy(trajectorySeedProducer)
globalMixedGSSeeds.firstHitSubDetectorNumber = [3]
globalMixedGSSeeds.firstHitSubDetectors = [1, 2, 6]
globalMixedGSSeeds.secondHitSubDetectorNumber = [3]
globalMixedGSSeeds.secondHitSubDetectors = [1, 2, 6]
globalMixedGSSeeds.seedingAlgo = ['GlobalMixed']

