import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
mixedTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
mixedTripletSeeds.numberOfHits = [3]
mixedTripletSeeds.firstHitSubDetectorNumber = [2]
mixedTripletSeeds.firstHitSubDetectors = [1, 2]
mixedTripletSeeds.secondHitSubDetectorNumber = [2]
mixedTripletSeeds.secondHitSubDetectors = [1, 2]
mixedTripletSeeds.thirdHitSubDetectorNumber = [3]
mixedTripletSeeds.thirdHitSubDetectors = [1, 2, 3]
mixedTripletSeeds.seedingAlgo = ['MixedTriplet']

