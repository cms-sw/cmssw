import FWCore.ParameterSet.Config as cms

slimmedElectrons = cms.EDProducer("PATElectronSlimmer",
   src = cms.InputTag("selectedPatElectrons"),
   dropSuperCluster = cms.bool(False),
   dropBasicClusters = cms.bool(False),
   dropPFlowClusters = cms.bool(False),
   dropPreshowerClusters = cms.bool(False),
   dropRecHits = cms.bool(False),
   linkToPackedPFCandidates = cms.bool(True),
   recoToPFMap = cms.InputTag("particleBasedIsolation","gedGsfElectrons"),
   packedPFCandidates = cms.InputTag("packedPFCandidates"), 
)

