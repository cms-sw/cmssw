import FWCore.ParameterSet.Config as cms

caloStage1LegacyFormatDigis = cms.EDProducer(
    "L1TCaloUpgradeToGCTConverter",
    InputCollection = cms.InputTag("caloStage1Digis"),
    InputRlxTauCollection = cms.InputTag("caloStage1Digis:rlxTaus"),
    InputIsoTauCollection = cms.InputTag("caloStage1Digis:isoTaus"),
    InputHFSumsCollection = cms.InputTag("caloStage1Digis:HFRingSums"),
    InputHFCountsCollection = cms.InputTag("caloStage1Digis:HFBitCounts")
)
