import FWCore.ParameterSet.Config as cms

import os

simGmtCaloSumDigis = cms.EDProducer('L1TMuonCaloSumProducer',
    caloStage2Layer2Label = cms.InputTag("simCaloStage2Layer1Digis"),
)

simGmtStage2Digis = cms.EDProducer('L1TMuonProducer',
    barrelTFInput  = cms.InputTag("simBmtfDigis", "BMTF"),
    overlapTFInput = cms.InputTag("simOmtfDigis", "OMTF"),
    forwardTFInput = cms.InputTag("simEmtfDigis", "EMTF"),
    #triggerTowerInput = cms.InputTag("simGmtCaloSumDigis", "TriggerTower2x2s"),
    triggerTowerInput = cms.InputTag("simGmtCaloSumDigis", "TriggerTowerSums"),
    autoBxRange = cms.bool(True), # if True the output BX range is calculated from the inputs and 'bxMin' and 'bxMax' are ignored
    bxMin = cms.int32(-2),
    bxMax = cms.int32(2)
)
