import FWCore.ParameterSet.Config as cms

l1tCaloStage2Layer1Digis = cms.EDProducer(
    "l1t::Stage2Layer1Producer",
    towerToken = cms.InputTag("l1tCaloStage2TowerDigis"),
    firmware = cms.int32(1)
)
