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
    autoCancelMode = cms.bool(True), # if True the cancel out methods are configured depending on the FW version number and 'bmtfCancelMode' + 'emtfCancelMode' are ignored
    bmtfCancelMode = cms.string("kftracks"), # 'tracks' or 'kftracks' (when using the Run-3 BMTF)
    emtfCancelMode = cms.string("coordinate") # 'tracks' or 'coordinate'
)

# Muon shower trigger
from L1Trigger.L1TMuon.simGmtShowerDigisDef_cfi import simGmtShowerDigisDef
simGmtShowerDigis = simGmtShowerDigisDef.clone()

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

## Era: Run2_2016
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(simGmtStage2Digis, barrelTFInput = cms.InputTag("simBmtfDigis", "BMTF"),
                                            autoCancelMode = cms.bool(False),
                                            bmtfCancelMode = cms.string("tracks"),
                                            emtfCancelMode = cms.string("coordinate"))

## Era: Run2_2017
from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017
stage2L1Trigger_2017.toModify(simGmtStage2Digis, barrelTFInput = cms.InputTag("simBmtfDigis", "BMTF"),
                                                 autoCancelMode = cms.bool(False),
                                                 bmtfCancelMode = cms.string("tracks"),
                                                 emtfCancelMode = cms.string("coordinate"))

### Era: Run2_2018
from Configuration.Eras.Modifier_stage2L1Trigger_2018_cff import stage2L1Trigger_2018
stage2L1Trigger_2018.toModify(simGmtStage2Digis, barrelTFInput = cms.InputTag("simBmtfDigis", "BMTF"),
                                                 autoCancelMode = cms.bool(False),
                                                 bmtfCancelMode = cms.string("tracks"),
                                                 emtfCancelMode = cms.string("coordinate"))

### Era: Run3_2021
from Configuration.Eras.Modifier_stage2L1Trigger_2021_cff import stage2L1Trigger_2021
stage2L1Trigger_2021.toModify(simGmtStage2Digis, barrelTFInput = cms.InputTag("simKBmtfDigis", "BMTF"),
                                                 autoCancelMode = cms.bool(False),
                                                 bmtfCancelMode = cms.string("kftracks"),
                                                 emtfCancelMode = cms.string("coordinate"))
