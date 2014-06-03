import FWCore.ParameterSet.Config as cms

caloStage2Digis = cms.EDProducer(
    "l1t::Stage2Layer2Producer",
    towerToken = cms.InputTag("caloStage2Layer1Digis"),
    firmware = cms.int32(1)
)
