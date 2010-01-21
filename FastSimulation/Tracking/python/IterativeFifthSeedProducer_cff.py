import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeFifthSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeFifthSeeds.firstHitSubDetectorNumber = [2]
iterativeFifthSeeds.firstHitSubDetectors = [5, 6]
iterativeFifthSeeds.secondHitSubDetectorNumber = [2]
iterativeFifthSeeds.secondHitSubDetectors = [5, 6]
iterativeFifthSeeds.thirdHitSubDetectorNumber = [0]
iterativeFifthSeeds.thirdHitSubDetectors = []
iterativeFifthSeeds.seedingAlgo = ['TobTecLayerPairs']
iterativeFifthSeeds.minRecHits = [4]
iterativeFifthSeeds.pTMin = [0.01]
#cut on fastsim simtracks. I think it should be removed for the 5th step
iterativeFifthSeeds.maxD0 = [99.]
iterativeFifthSeeds.maxZ0 = [99.]
#-----
iterativeFifthSeeds.numberOfHits = [2]
#values for the seed compatibility constraint
iterativeFifthSeeds.originRadius = [5.0]
iterativeFifthSeeds.originHalfLength = [10.0]
###iterativeFifthSeeds.originpTMin = [0.8]
iterativeFifthSeeds.originpTMin = [0.5]
iterativeFifthSeeds.zVertexConstraint = [-1.0]
iterativeFifthSeeds.primaryVertices = ['none']

