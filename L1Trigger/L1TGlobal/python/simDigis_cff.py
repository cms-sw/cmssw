#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
import FWCore.ParameterSet.Config as cms
import sys
#
# Legacy Trigger:
#
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
if not (stage2L1Trigger.isChosen()):
#
# -  Global Trigger emulator
#
    import L1Trigger.GlobalTrigger.gtDigis_cfi
    simGtDigis = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone()
    simGtDigis.GmtInputTag = 'simGmtDigis'
    simGtDigis.GctInputTag = 'simGctDigis'
    simGtDigis.TechnicalTriggersInputTags = cms.VInputTag(
        cms.InputTag( 'simBscDigis' ), 
        cms.InputTag( 'simRpcTechTrigDigis' ),
        cms.InputTag( 'simHcalTechTrigDigis' ),
        cms.InputTag( 'simCastorTechTrigDigis' )
        )
    SimL1TGlobal = cms.Sequence(simGtDigis)

#
# Stage-2 Trigger
#
if stage2L1Trigger.isChosen():
#
# -  Global Trigger emulator
#
    from L1Trigger.L1TGlobal.simGtStage2Digis_cfi import *
    SimL1TGlobal = cms.Sequence(simGtStage2Digis)
