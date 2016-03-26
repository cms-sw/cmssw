import FWCore.ParameterSet.Config as cms

import os

simGmtCaloSumDigis = cms.EDProducer('L1TMuonCaloSumProducer',
    caloStage2Layer2Label = cms.InputTag("simCaloStage2Layer1Digis"),
)

simGmtStage2Digis = cms.EDProducer('L1TMuonProducer',
    barrelTFInput  = cms.InputTag("simBmtfDigis", "BMTF"),
    overlapTFInput = cms.InputTag("simOmtfDigis", "OMTF"),
    forwardTFInput = cms.InputTag("simEmtfDigis", "EMTF"),
    #barrelTFInput  = cms.InputTag("simMuonQualityAdjusterDigis", "BMTF"),
    #overlapTFInput = cms.InputTag("simMuonQualityAdjusterDigis", "OMTF"),
    #forwardTFInput = cms.InputTag("simMuonQualityAdjusterDigis", "EMTF"),
    #triggerTowerInput = cms.InputTag("simGmtCaloSumDigis", "TriggerTower2x2s"),
    triggerTowerInput = cms.InputTag("simGmtCaloSumDigis", "TriggerTowerSums"),
)
