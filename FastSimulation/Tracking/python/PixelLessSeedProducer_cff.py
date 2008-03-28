import FWCore.ParameterSet.Config as cms

import copy
from FastSimulation.Tracking.TrajectorySeedProducer_cfi import *
pixelLessGSSeeds = copy.deepcopy(trajectorySeedProducer)
pixelLessGSSeeds.firstHitSubDetectorNumber = [3]
pixelLessGSSeeds.firstHitSubDetectors = [3, 4, 6]
pixelLessGSSeeds.secondHitSubDetectorNumber = [3]
pixelLessGSSeeds.secondHitSubDetectors = [3, 4, 6]
pixelLessGSSeeds.seedingAlgo = ['PixelLess']

