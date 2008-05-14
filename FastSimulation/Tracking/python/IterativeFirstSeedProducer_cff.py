import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeFirstSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeFirstSeeds.firstHitSubDetectorNumber = [1, 3]
iterativeFirstSeeds.firstHitSubDetectors = [1, 1, 2, 6]
iterativeFirstSeeds.secondHitSubDetectorNumber = [2, 3]
iterativeFirstSeeds.secondHitSubDetectors = [1, 2, 1, 2, 6]
iterativeFirstSeeds.thirdHitSubDetectorNumber = [3, 0]
iterativeFirstSeeds.thirdHitSubDetectors = [1, 2, 3]
iterativeFirstSeeds.seedingAlgo = ['FirstMixedTriplets', 'FirstMixedPairs']
iterativeFirstSeeds.minRecHits = [5, 5]
iterativeFirstSeeds.pTMin = [0.3, 0.3]
iterativeFirstSeeds.maxD0 = [1., 1.]
iterativeFirstSeeds.maxZ0 = [30., 30.]
iterativeFirstSeeds.numberOfHits = [3, 2]
iterativeFirstSeeds.originRadius = [0.2, 0.2]
iterativeFirstSeeds.originHalfLength = [15.9, 15.9]
iterativeFirstSeeds.originpTMin = [0.9, 0.9]
iterativeFirstSeeds.zVertexConstraint = [-1.0, 0.4]
iterativeFirstSeeds.primaryVertices = ['none', 'pixelVertices']

