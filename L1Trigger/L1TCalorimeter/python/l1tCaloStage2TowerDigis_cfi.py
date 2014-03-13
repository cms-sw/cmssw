import FWCore.ParameterSet.Config as cms

l1tCaloStage2TowerDigis = cms.EDProducer(
    "l1t::L1TCaloTowerProducer",
    verbosity = cms.int32(2),
    ecalToken = cms.InputTag("ecalDigis"),
    hcalToken = cms.InputTag("hcalDigis")
)
