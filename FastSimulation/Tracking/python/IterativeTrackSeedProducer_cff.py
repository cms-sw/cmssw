import FWCore.ParameterSet.Config as cms

import copy
from FastSimulation.Tracking.TrajectorySeedProducer_cfi import *
iterativeTrackingSeeds = copy.deepcopy(trajectorySeedProducer)
iterativeTrackingSeeds.firstHitSubDetectorNumber = [1, 3, 2, 3, 3]
iterativeTrackingSeeds.firstHitSubDetectors = [1, 1, 2, 6, 1, 
    3, 1, 2, 6, 1, 
    2, 6]
iterativeTrackingSeeds.secondHitSubDetectorNumber = [2, 3, 3, 3, 3]
iterativeTrackingSeeds.secondHitSubDetectors = [1, 2, 1, 2, 6, 
    1, 2, 3, 1, 2, 
    6, 1, 2, 6]
iterativeTrackingSeeds.thirdHitSubDetectorNumber = [3, 0, 3, 0, 0]
iterativeTrackingSeeds.thirdHitSubDetectors = [1, 2, 3, 1, 2, 
    3]
iterativeTrackingSeeds.seedingAlgo = ['FirstMixedTriplets', 'FirstMixedPairs', 'SecondMixedTriplets', 'SecondMixedPairs', 'ThirdMixedPairs']
iterativeTrackingSeeds.minRecHits = [5, 5, 3, 3, 3]
iterativeTrackingSeeds.pTMin = [0.3, 0.3, 0.3, 0.3, 0.3]
iterativeTrackingSeeds.maxD0 = [1., 1., 1., 1., 1.]
iterativeTrackingSeeds.maxZ0 = [30., 30., 30., 30., 30.]
iterativeTrackingSeeds.numberOfHits = [3, 2, 3, 2, 2]
iterativeTrackingSeeds.originRadius = [0.2, 0.2, 0.2, 0.2, 0.2]
iterativeTrackingSeeds.originHalfLength = [15.9, 15.9, 15.9, 15.9, 15.9]
iterativeTrackingSeeds.originpTMin = [0.9, 0.9, 0.3, 0.3, 0.3]
iterativeTrackingSeeds.zVertexConstraint = [-1.0, 0.4, -1.0, 0.4, -1.0]
iterativeTrackingSeeds.primaryVertices = ['none', 'pixelVertices', 'none', 'pixelVertices', 'none']

