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
#
# -  Global Trigger emulator
#
import L1Trigger.GlobalTrigger.gtDigis_cfi
simGtDigis = L1Trigger.GlobalTrigger.gtDigis_cfi.gtDigis.clone(
    GmtInputTag = 'simGmtDigis',
    GctInputTag = 'simGctDigis',
    TechnicalTriggersInputTags = [
        'simBscDigis',
        'simRpcTechTrigDigis',
        'simHcalTechTrigDigis',
        'simCastorTechTrigDigis'
    ]
)
SimL1TGlobalTask = cms.Task(simGtDigis)
SimL1TGlobal = cms.Sequence(SimL1TGlobalTask)

#
# Stage-2 Trigger
#
#
# -  Global Trigger emulator
#
from L1Trigger.L1TGlobal.simGtStage2Digis_cfi import *
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toReplaceWith(SimL1TGlobalTask, cms.Task(simGtStage2Digis))
