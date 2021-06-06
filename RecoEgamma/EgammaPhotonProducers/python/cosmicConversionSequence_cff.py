import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.conversionSequence_cff import *
conversions.scHybridBarrelProducer = "cosmicSuperClusters:CosmicBarrelSuperClusters"
conversions.scIslandEndcapProducer = "cosmicSuperClusters:CosmicEndcapSuperClusters"
conversions.bcBarrelCollection     = "cosmicBasicClusters:CosmicBarrelBasicClusters"
conversions.bcEndcapCollection     = "cosmicBasicClusters:CosmicEndcapBasicClusters"
conversions.recoverOneTrackCase    = False

cosmicConversionTask = cms.Task(conversionTask)
cosmicConversionSequence = cms.Sequence(cosmicConversionTask)
