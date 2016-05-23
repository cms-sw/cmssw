import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

#
# Legacy Trigger:
#
if not (eras.stage1L1Trigger.isChosen() or eras.stage2L1Trigger.isChosen()):
    print "L1TCalorimeter Sequence configured for Run1 (Legacy) trigger. "
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
if eras.stage1L1Trigger.isChosen() and not eras.stage2L1Trigger.isChosen():
    print "L1TCalorimeter Sequence configured for Stage-1 (2015) trigger. "    
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
if eras.stage2L1Trigger.isChosen():
    from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import *
    print "L1TCalorimeter Sequence configured for Stage-2 (2016) trigger. "
    # select one of the following two options:
    # - layer1 from L1Trigger/L1TCalorimeter package
    #from L1Trigger.L1TCalorimeter.simCaloStage2Layer1Digis_cfi import simCaloStage2Layer1Digis
    # - layer1 from L1Trigger/L1TCaloLayer1 package
    from L1Trigger.L1TCaloLayer1.simCaloStage2Layer1Digis_cfi import simCaloStage2Layer1Digis
    from L1Trigger.L1TCalorimeter.simCaloStage2Digis_cfi import simCaloStage2Digis
    SimL1TCalorimeter = cms.Sequence( simCaloStage2Layer1Digis + simCaloStage2Digis )

    
