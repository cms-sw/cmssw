import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeFirstSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeFirstSeeds.firstHitSubDetectorNumber = [1, 3]
iterativeFirstSeeds.firstHitSubDetectors = [1, 1, 2,6]
iterativeFirstSeeds.secondHitSubDetectorNumber = [2, 3]
iterativeFirstSeeds.secondHitSubDetectors = [1, 2, 1, 2, 6]
iterativeFirstSeeds.thirdHitSubDetectorNumber = [2, 0]
iterativeFirstSeeds.thirdHitSubDetectors = [1, 2]
iterativeFirstSeeds.seedingAlgo = ['FirstPixelTriplets', 'FirstMixedPairs']
iterativeFirstSeeds.minRecHits = [3, 3]
##iterativeFirstSeeds.pTMin = [0.35, 0.35]
iterativeFirstSeeds.pTMin = [0.6, 0.6]
iterativeFirstSeeds.maxD0 = [1., 1.]
iterativeFirstSeeds.maxZ0 = [30., 30.]
iterativeFirstSeeds.numberOfHits = [3, 2]
###iterativeFirstSeeds.originRadius = [0.2, 0.2] FirstStep rev 1.12!!!!
iterativeFirstSeeds.originRadius = [0.2, 0.05]
iterativeFirstSeeds.originHalfLength = [15.9, 15.9]
###iterativeFirstSeeds.originpTMin = [0.5, 0.9]
iterativeFirstSeeds.originpTMin = [0.8, 0.7]
##iterativeFirstSeeds.zVertexConstraint = [-1.0, 0.4]
iterativeFirstSeeds.zVertexConstraint = [-1.0, 0.8]
iterativeFirstSeeds.primaryVertices = ['none', 'pixelVertices']

