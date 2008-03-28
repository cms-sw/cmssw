import FWCore.ParameterSet.Config as cms

import copy
from FastSimulation.Tracking.TrajectorySeedProducer_cfi import *
pixelTripletGSSeeds = copy.deepcopy(trajectorySeedProducer)
pixelTripletGSSeeds.numberOfHits = [3]
pixelTripletGSSeeds.firstHitSubDetectorNumber = [2]
pixelTripletGSSeeds.firstHitSubDetectors = [1, 2]
pixelTripletGSSeeds.secondHitSubDetectorNumber = [2]
pixelTripletGSSeeds.secondHitSubDetectors = [1, 2]
pixelTripletGSSeeds.thirdHitSubDetectorNumber = [2]
pixelTripletGSSeeds.thirdHitSubDetectors = [1, 2]
pixelTripletGSSeeds.seedingAlgo = ['PixelTriplet']

