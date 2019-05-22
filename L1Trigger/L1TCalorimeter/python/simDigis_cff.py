import FWCore.ParameterSet.Config as cms
import sys

#
# Legacy Trigger:
#
# -  RCT (Regional Calorimeter Trigger) emulator
import L1Trigger.RegionalCaloTrigger.rctDigis_cfi
simRctDigis = L1Trigger.RegionalCaloTrigger.rctDigis_cfi.rctDigis.clone(
    ecalDigis = ['simEcalTriggerPrimitiveDigis'],
    hcalDigis = ['simHcalTriggerPrimitiveDigis']
)
# - GCT (Global Calorimeter Trigger) emulator
import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
simGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone(
    inputLabel = 'simRctDigis'
)
SimL1TCalorimeterTask = cms.Task(simRctDigis, simGctDigis)
SimL1TCalorimeter = cms.Sequence(SimL1TCalorimeterTask)

#
# Stage-1 Trigger
#
#
# - Stage-1 Layer-2 Calorimeter Trigger Emulator, with required converters (Stage-1 mixes legacy and upgrade) 
#
from L1Trigger.L1TCalorimeter.simRctUpgradeFormatDigis_cfi import *
from L1Trigger.L1TCalorimeter.simCaloStage1Digis_cfi import *
from L1Trigger.L1TCalorimeter.simCaloStage1FinalDigis_cfi import *
from L1Trigger.L1TCalorimeter.simCaloStage1LegacyFormatDigis_cfi import *
from L1Trigger.L1TCalorimeter.caloConfigStage1PP_cfi import *
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
(stage1L1Trigger & ~stage2L1Trigger).toReplaceWith(SimL1TCalorimeterTask, cms.Task(simRctDigis, simRctUpgradeFormatDigis, simCaloStage1Digis, simCaloStage1FinalDigis, simCaloStage1LegacyFormatDigis))

#
# Stage-2 Trigger
#
# select one of the following two options:
# - layer1 from L1Trigger/L1TCalorimeter package
#from L1Trigger.L1TCalorimeter.simCaloStage2Layer1Digis_cfi import simCaloStage2Layer1Digis
# - layer1 from L1Trigger/L1TCaloLayer1 package
from L1Trigger.L1TCaloLayer1.simCaloStage2Layer1Digis_cfi import simCaloStage2Layer1Digis
from L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi import simCaloStage2Digis
stage2L1Trigger.toReplaceWith(SimL1TCalorimeterTask, cms.Task( simCaloStage2Layer1Digis, simCaloStage2Digis ))

def _modifyStage2L1TriggerCaloParams(process):
    from CondCore.CondDB.CondDB_cfi import CondDB
    CondDB.connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
    process.l1conddb = cms.ESSource("PoolDBESSource",
       CondDB,
       toGet   = cms.VPSet(
            cms.PSet(
                 record = cms.string('L1TCaloParamsO2ORcd'),
                 tag = cms.string("L1TCaloParams_static_CMSSW_9_2_10_2017_v1_8_2_updateHFSF_v6MET")
            )
       )
    )
modifySimDigis_stage2L1TriggerCaloPArams = stage2L1Trigger.makeProcessModifier(_modifyStage2L1TriggerCaloParams)
