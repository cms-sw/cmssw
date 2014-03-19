import FWCore.ParameterSet.Config as cms

l1tCaloStage2Digis = cms.EDProducer(
    "l1t::L1TCaloStage2Producer",
    towerToken = cms.InputTag("l1tCaloStage2Layer1Digis"),
    firmware = cms.int32(1)
)
