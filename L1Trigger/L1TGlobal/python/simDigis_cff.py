#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
import FWCore.ParameterSet.Config as cms
#
# Legacy Trigger:
#
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
if not (stage2L1Trigger.isChosen() or phase2_common.isChosen()):
    print "L1TGlobal Sequence configured for Legacy trigger (Run1 and Run 2015). "
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
if (stage2L1Trigger.isChosen() or phase2_common.isChosen()):
#
# -  Global Trigger emulator
#
    print "L1TGlobal Sequence configured for Stage-2 (2016) trigger. "
    from L1Trigger.L1TGlobal.simGtStage2Digis_cfi import *
    SimL1TGlobal = cms.Sequence(simGtStage2Digis)
