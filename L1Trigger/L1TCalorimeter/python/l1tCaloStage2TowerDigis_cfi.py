import FWCore.ParameterSet.Config as cms

l1tCaloStage2TowerDigis = cms.EDProducer(
    "L1TCaloTowerProducer",
    ecalToken = cms.InputTag("ecalDigis"),
    hcalToken = cms.InputTag("hcalDigis")
)
