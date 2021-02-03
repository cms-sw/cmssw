import FWCore.ParameterSet.Config as cms

caloStage1FinalDigis = cms.EDProducer("L1TPhysicalEtAdder",
    InputCollection = cms.InputTag("caloStage1Digis"),
    InputHFCountsCollection = cms.InputTag("caloStage1Digis","HFBitCounts"),
    InputHFSumsCollection = cms.InputTag("caloStage1Digis","HFRingSums"),
    InputIsoTauCollection = cms.InputTag("caloStage1Digis","isoTaus"),
    InputPreGtJetCollection = cms.InputTag("caloStage1Digis"),
    InputRlxTauCollection = cms.InputTag("caloStage1Digis","rlxTaus")
)
