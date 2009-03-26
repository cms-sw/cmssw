import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrajectorySeedProducer_cfi
iterativeFourthSeeds = FastSimulation.Tracking.TrajectorySeedProducer_cfi.trajectorySeedProducer.clone()
iterativeFourthSeeds.firstHitSubDetectorNumber = [3]
iterativeFourthSeeds.firstHitSubDetectors = [3, 4, 6]
iterativeFourthSeeds.secondHitSubDetectorNumber = [3]
iterativeFourthSeeds.secondHitSubDetectors = [3, 4, 6]
iterativeFourthSeeds.thirdHitSubDetectorNumber = [0]
iterativeFourthSeeds.thirdHitSubDetectors = []
iterativeFourthSeeds.seedingAlgo = ['FourthPixelLessPairs']
iterativeFourthSeeds.minRecHits = [5]
iterativeFourthSeeds.pTMin = [0.2]
#cut on fastsim simtracks. I think it should be removed for the 4th step
iterativeFourthSeeds.maxD0 = [10.]
iterativeFourthSeeds.maxZ0 = [30.]
#-----
iterativeFourthSeeds.numberOfHits = [2]
#values for the seed compatibility constraint
iterativeFourthSeeds.originRadius = [2.0]
iterativeFourthSeeds.originHalfLength = [10.0]
iterativeFourthSeeds.originpTMin = [0.5]
iterativeFourthSeeds.zVertexConstraint = [-1.0]
iterativeFourthSeeds.primaryVertices = ['none']

