import FWCore.ParameterSet.Config as cms

import os

simGmtCaloSumDigis = cms.EDProducer('L1TMuonCaloSumProducer',
    caloStage2Layer2Label = cms.InputTag("simCaloStage2Layer1Digis"),
)

simGmtStage2Digis = cms.EDProducer('L1TMuonProducer',
    barrelTFInput  = cms.InputTag("simKBmtfDigis", "BMTF"),
    overlapTFInput = cms.InputTag("simOmtfDigis", "OMTF"),
    forwardTFInput = cms.InputTag("simEmtfDigis", "EMTF"),
    #triggerTowerInput = cms.InputTag("simGmtCaloSumDigis", "TriggerTower2x2s"),
    triggerTowerInput = cms.InputTag("simGmtCaloSumDigis", "TriggerTowerSums"),
    autoBxRange = cms.bool(True), # if True the output BX range is calculated from the inputs and 'bxMin' and 'bxMax' are ignored
    bxMin = cms.int32(-2),
    bxMax = cms.int32(2),
    autoCancelMode = cms.bool(False), # if True the cancel out methods are configured depending on the FW version number and 'emtfCancelMode' is ignored
    emtfCancelMode = cms.string("coordinate"), # 'tracks' or 'coordinate'
    runPhase2 = cms.bool(False)
)

from CondCore.CondDB.CondDB_cfi import CondDB
CondDB.connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
l1ugmtdb = cms.ESSource("PoolDBESSource",
       CondDB,
       toGet   = cms.VPSet(
            cms.PSet(
                 record = cms.string('L1TMuonGlobalParamsO2ORcd'),
                 tag = cms.string("L1TMuonGlobalParamsPrototype_Stage2v0_hlt")
            )
       )
)

from Configuration.Eras.Modifier_phase2_muon_cff import phase2_muon
phase2_muon.toModify( simGmtStage2Digis, runPhase2 =cms.bool(True) )
