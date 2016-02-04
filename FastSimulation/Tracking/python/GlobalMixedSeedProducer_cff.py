import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
globalMixedSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
globalMixedSeeds.firstHitSubDetectorNumber = [3]
globalMixedSeeds.firstHitSubDetectors = [1, 2, 6]
globalMixedSeeds.secondHitSubDetectorNumber = [3]
globalMixedSeeds.secondHitSubDetectors = [1, 2, 6]
globalMixedSeeds.seedingAlgo = ['GlobalMixed']

