import FWCore.ParameterSet.Config as cms

simCaloStage1FinalDigis = cms.EDProducer(
    "L1TPhysicalEtAdder",
    InputCollection = cms.InputTag("simCaloStage1Digis"),
    InputRlxTauCollection = cms.InputTag("simCaloStage1Digis:rlxTaus"),
    InputIsoTauCollection = cms.InputTag("simCaloStage1Digis:isoTaus"),
    InputPreGtJetCollection = cms.InputTag("simCaloStage1Digis:preGtJets"),
    InputHFSumsCollection = cms.InputTag("simCaloStage1Digis:HFRingSums"),
    InputHFCountsCollection = cms.InputTag("simCaloStage1Digis:HFBitCounts")
)
