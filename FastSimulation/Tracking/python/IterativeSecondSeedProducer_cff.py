import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeSecondSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeSecondSeeds.firstHitSubDetectorNumber = [2, 3]
iterativeSecondSeeds.firstHitSubDetectors = [1, 3, 1, 2, 6]
iterativeSecondSeeds.secondHitSubDetectorNumber = [3, 3]
iterativeSecondSeeds.secondHitSubDetectors = [1, 2, 3, 1, 2, 
    6]
iterativeSecondSeeds.thirdHitSubDetectorNumber = [3, 0]
iterativeSecondSeeds.thirdHitSubDetectors = [1, 2, 3]
iterativeSecondSeeds.seedingAlgo = ['SecondMixedTriplets', 'SecondMixedPairs']
iterativeSecondSeeds.minRecHits = [3, 3]
iterativeSecondSeeds.pTMin = [0.3, 0.3]
iterativeSecondSeeds.maxD0 = [1., 1.]
iterativeSecondSeeds.maxZ0 = [30., 30.]
iterativeSecondSeeds.numberOfHits = [3, 2]
iterativeSecondSeeds.originRadius = [0.2, 0.2]
iterativeSecondSeeds.originHalfLength = [15.9, 15.9]
iterativeSecondSeeds.originpTMin = [0.3, 0.3]
iterativeSecondSeeds.zVertexConstraint = [-1.0, 0.4]
iterativeSecondSeeds.primaryVertices = ['none', 'pixelVertices']

