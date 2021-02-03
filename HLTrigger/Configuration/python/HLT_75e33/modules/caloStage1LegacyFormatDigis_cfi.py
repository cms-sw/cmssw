import FWCore.ParameterSet.Config as cms

caloStage1LegacyFormatDigis = cms.EDProducer("L1TCaloUpgradeToGCTConverter",
    InputCollection = cms.InputTag("caloStage1Digis"),
    InputHFCountsCollection = cms.InputTag("caloStage1Digis","HFBitCounts"),
    InputHFSumsCollection = cms.InputTag("caloStage1Digis","HFRingSums"),
    InputIsoTauCollection = cms.InputTag("caloStage1Digis","isoTaus"),
    InputRlxTauCollection = cms.InputTag("caloStage1Digis","rlxTaus")
)
