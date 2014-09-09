import FWCore.ParameterSet.Config as cms

l1tCaloStage2TowerDigis = cms.EDProducer(
    "l1t::L1TCaloTowerProducer",
    verbosity = cms.int32(2),
    bxFirst    = cms.int32(0),
    bxLast     = cms.int32(0),
    ecalToken = cms.InputTag("ecalDigis:EcalTriggerPrimitives"),
    hcalToken = cms.InputTag("hcalDigis")
)
