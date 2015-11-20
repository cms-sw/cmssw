import FWCore.ParameterSet.Config as cms

import os

simGmtCaloSumDigis = cms.EDProducer('L1TMuonCaloSumProducer',
    caloStage2Layer2Label = cms.InputTag("caloStage2Layer1Digis"),
)

simGmtDigis = cms.EDProducer('L1TMuonProducer',
    barrelTFInput  = cms.InputTag("simBmtfDigis", "BMTF"),
    overlapTFInput = cms.InputTag("simOmtfDigis", "OMTF"),
    forwardTFInput = cms.InputTag("simEmtfDigis", "EMTF"),
    triggerTowerInput = cms.InputTag("simGmtCaloSumDigis", "TriggerTower2x2s"),
    #triggerTowerInput = cms.InputTag("simGmtCaloSumDigis", "TriggerTowerSums"),
)

