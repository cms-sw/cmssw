import FWCore.ParameterSet.Config as cms

import copy
from FastSimulation.Tracking.TrajectorySeedProducer_cfi import *
mixedTripletGSSeeds = copy.deepcopy(trajectorySeedProducer)
mixedTripletGSSeeds.numberOfHits = [3]
mixedTripletGSSeeds.firstHitSubDetectorNumber = [2]
mixedTripletGSSeeds.firstHitSubDetectors = [1, 2]
mixedTripletGSSeeds.secondHitSubDetectorNumber = [2]
mixedTripletGSSeeds.secondHitSubDetectors = [1, 2]
mixedTripletGSSeeds.thirdHitSubDetectorNumber = [3]
mixedTripletGSSeeds.thirdHitSubDetectors = [1, 2, 3]
mixedTripletGSSeeds.seedingAlgo = ['MixedTriplet']

