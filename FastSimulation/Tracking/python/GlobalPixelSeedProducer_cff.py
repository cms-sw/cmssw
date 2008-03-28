import FWCore.ParameterSet.Config as cms

import copy
from FastSimulation.Tracking.TrajectorySeedProducer_cfi import *
globalPixelGSSeeds = copy.deepcopy(trajectorySeedProducer)
globalPixelGSSeeds.firstHitSubDetectorNumber = [2]
globalPixelGSSeeds.firstHitSubDetectors = [1, 2]
globalPixelGSSeeds.secondHitSubDetectorNumber = [2]
globalPixelGSSeeds.secondHitSubDetectors = [1, 2]
globalPixelGSSeeds.seedingAlgo = ['GlobalPixel']

