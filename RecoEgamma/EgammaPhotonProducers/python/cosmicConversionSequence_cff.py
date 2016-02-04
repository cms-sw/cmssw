import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaPhotonProducers.conversionSequence_cff import *
conversions.scHybridBarrelProducer = cms.InputTag("cosmicSuperClusters","CosmicBarrelSuperClusters")
conversions.scIslandEndcapProducer = cms.InputTag("cosmicSuperClusters","CosmicEndcapSuperClusters")
conversions.bcBarrelCollection = cms.InputTag("cosmicBasicClusters","CosmicBarrelBasicClusters")
conversions.bcEndcapCollection = cms.InputTag("cosmicBasicClusters","CosmicEndcapBasicClusters")

conversionTrackCandidates.scHybridBarrelProducer = cms.InputTag("cosmicSuperClusters","CosmicBarrelSuperClusters")
conversionTrackCandidates.scIslandEndcapProducer = cms.InputTag("cosmicSuperClusters","CosmicEndcapSuperClusters")
conversionTrackCandidates.bcBarrelCollection = cms.InputTag("cosmicBasicClusters","CosmicBarrelBasicClusters")
conversionTrackCandidates.bcEndcapCollection = cms.InputTag("cosmicBasicClusters","CosmicEndcapBasicClusters")
conversions.recoverOneTrackCase =  cms.bool(False)

cosmicConversionSequence = cms.Sequence(conversionSequence)
