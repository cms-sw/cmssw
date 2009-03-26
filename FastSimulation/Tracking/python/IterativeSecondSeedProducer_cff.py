import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeSecondSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeSecondSeeds.firstHitSubDetectorNumber = [1]
iterativeSecondSeeds.firstHitSubDetectors = [1]
iterativeSecondSeeds.secondHitSubDetectorNumber = [2]
iterativeSecondSeeds.secondHitSubDetectors = [1, 2]
iterativeSecondSeeds.thirdHitSubDetectorNumber = [2]
iterativeSecondSeeds.thirdHitSubDetectors = [1, 2]
iterativeSecondSeeds.seedingAlgo = ['SecondPixelTriplets']
iterativeSecondSeeds.minRecHits = [3]
iterativeSecondSeeds.pTMin = [0.1]
iterativeSecondSeeds.maxD0 = [1.]
iterativeSecondSeeds.maxZ0 = [30.]
iterativeSecondSeeds.numberOfHits = [3]
iterativeSecondSeeds.originRadius = [0.2]
iterativeSecondSeeds.originHalfLength = [17.5]
iterativeSecondSeeds.originpTMin = [0.2]
iterativeSecondSeeds.zVertexConstraint = [-1.0]
iterativeSecondSeeds.primaryVertices = ['none']


