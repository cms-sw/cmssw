import FWCore.ParameterSet.Config as cms
import sys

#
# Legacy Trigger:
#
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
if not (stage1L1Trigger.isChosen() or stage2L1Trigger.isChosen()):
# -  RCT (Regional Calorimeter Trigger) emulator
    import L1Trigger.RegionalCaloTrigger.rctDigis_cfi
    simRctDigis = L1Trigger.RegionalCaloTrigger.rctDigis_cfi.rctDigis.clone()
    simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'simEcalTriggerPrimitiveDigis' ) )
    simRctDigis.hcalDigis = cms.VInputTag( cms.InputTag( 'simHcalTriggerPrimitiveDigis' ) )
# - GCT (Global Calorimeter Trigger) emulator
    import L1Trigger.GlobalCaloTrigger.gctDigis_cfi
    simGctDigis = L1Trigger.GlobalCaloTrigger.gctDigis_cfi.gctDigis.clone()
    simGctDigis.inputLabel = 'simRctDigis'
    SimL1TCalorimeter = cms.Sequence(simRctDigis + simGctDigis)

#
# Stage-1 Trigger
#
if stage1L1Trigger.isChosen() and not stage2L1Trigger.isChosen():
#
# -  RCT (Regional Calorimeter Trigger) emulator
#
    import L1Trigger.RegionalCaloTrigger.rctDigis_cfi
    simRctDigis = L1Trigger.RegionalCaloTrigger.rctDigis_cfi.rctDigis.clone()
    simRctDigis.ecalDigis = cms.VInputTag( cms.InputTag( 'simEcalTriggerPrimitiveDigis' ) )
    simRctDigis.hcalDigis = cms.VInputTag( cms.InputTag( 'simHcalTriggerPrimitiveDigis' ) )
#
# - Stage-1 Layer-2 Calorimeter Trigger Emulator, with required converters (Stage-1 mixes legacy and upgrade) 
#
    from L1Trigger.L1TCalorimeter.simRctUpgradeFormatDigis_cfi import *
    from L1Trigger.L1TCalorimeter.simCaloStage1Digis_cfi import *
    from L1Trigger.L1TCalorimeter.simCaloStage1FinalDigis_cfi import *
    from L1Trigger.L1TCalorimeter.simCaloStage1LegacyFormatDigis_cfi import *
    SimL1TCalorimeter = cms.Sequence(simRctDigis + simRctUpgradeFormatDigis + simCaloStage1Digis + simCaloStage1FinalDigis + simCaloStage1LegacyFormatDigis)
#
# Stage-2 Trigger
#
if stage2L1Trigger.isChosen():
    # select one of the following two options:
    # - layer1 from L1Trigger/L1TCalorimeter package
    #from L1Trigger.L1TCalorimeter.simCaloStage2Layer1Digis_cfi import simCaloStage2Layer1Digis
    # - layer1 from L1Trigger/L1TCaloLayer1 package
    from L1Trigger.L1TCaloLayer1.simCaloStage2Layer1Digis_cfi import simCaloStage2Layer1Digis
    from L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi import simCaloStage2Digis
    SimL1TCalorimeter = cms.Sequence( simCaloStage2Layer1Digis + simCaloStage2Digis )

    
