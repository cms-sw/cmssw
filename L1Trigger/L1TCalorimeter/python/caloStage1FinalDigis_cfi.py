import FWCore.ParameterSet.Config as cms

caloStage1FinalDigis = cms.EDProducer(
    "L1TPhysicalEtAdder",
    InputCollection = cms.InputTag("caloStage1Digis"),
    InputRlxTauCollection = cms.InputTag("caloStage1Digis:rlxTaus"),
    InputIsoTauCollection = cms.InputTag("caloStage1Digis:isoTaus"),
    InputPreGtJetCollection = cms.InputTag("caloStage1Digis:preGtJets"),
    InputHFSumsCollection = cms.InputTag("caloStage1Digis:HFRingSums"),
    InputHFCountsCollection = cms.InputTag("caloStage1Digis:HFBitCounts")
)
