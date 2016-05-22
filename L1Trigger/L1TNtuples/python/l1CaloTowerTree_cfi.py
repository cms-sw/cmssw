import FWCore.ParameterSet.Config as cms

from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *

l1CaloTowerTree = cms.EDAnalyzer(
    "L1CaloTowerTreeProducer",
    ecalToken = cms.untracked.InputTag("ecalDigis:EcalTriggerPrimitives"),
    hcalToken = cms.untracked.InputTag("hcalDigis"),
    l1TowerToken = cms.untracked.InputTag("caloStage2Digis","CaloTower"),
    l1ClusterToken = cms.untracked.InputTag("")
)

