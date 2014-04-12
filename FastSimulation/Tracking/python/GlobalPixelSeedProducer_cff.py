import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
globalPixelSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
globalPixelSeeds.firstHitSubDetectorNumber = [2]
globalPixelSeeds.firstHitSubDetectors = [1, 2]
globalPixelSeeds.secondHitSubDetectorNumber = [2]
globalPixelSeeds.secondHitSubDetectors = [1, 2]
globalPixelSeeds.seedingAlgo = ['GlobalPixel']

