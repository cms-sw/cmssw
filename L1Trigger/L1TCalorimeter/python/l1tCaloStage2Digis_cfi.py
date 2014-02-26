import FWCore.ParameterSet.Config as cms

l1tStage2CaloDigis = cms.EDProducer(
    "L1TCaloStage2Producer",
    towerToken = cms.InputTag("l1tCaloStage2TowerDigis")
)
