import FWCore.ParameterSet.Config as cms

caloStage1FinalDigis = cms.EDProducer(
    "l1t::PhysicalEtAdder",
    InputCollection = cms.InputTag("caloStage1Digis"),
    InputRlxTauCollection = cms.InputTag("caloStage1Digis:rlxTaus"),
    InputIsoTauCollection = cms.InputTag("caloStage1Digis:isoTaus")
)


