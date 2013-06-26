import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelTripletSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
pixelTripletSeeds.numberOfHits = [3]
pixelTripletSeeds.firstHitSubDetectorNumber = [2]
pixelTripletSeeds.firstHitSubDetectors = [1, 2]
pixelTripletSeeds.secondHitSubDetectorNumber = [2]
pixelTripletSeeds.secondHitSubDetectors = [1, 2]
pixelTripletSeeds.thirdHitSubDetectorNumber = [2]
pixelTripletSeeds.thirdHitSubDetectors = [1, 2]
pixelTripletSeeds.seedingAlgo = ['PixelTriplet']

