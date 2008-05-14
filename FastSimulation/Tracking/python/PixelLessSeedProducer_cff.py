import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
pixelLessSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
pixelLessSeeds.firstHitSubDetectorNumber = [3]
pixelLessSeeds.firstHitSubDetectors = [3, 4, 6]
pixelLessSeeds.secondHitSubDetectorNumber = [3]
pixelLessSeeds.secondHitSubDetectors = [3, 4, 6]
pixelLessSeeds.seedingAlgo = ['PixelLess']

