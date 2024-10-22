#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
# hackConditions.py  Load ES Producers for any conditions not yet in GT...
#
# The intention is that this file should shrink with time as conditions are added to GT.
#

import FWCore.ParameterSet.Config as cms
import sys

#
# Legacy Trigger:  No Hacks Needed
#
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger

#
# Stage-1 Trigger:  No Hacks Needed
#

#
# Stage-2 Trigger
#
def _load(process, fs):
    for f in fs:
        process.load(f)
    #process.TriggerMenu.L1TriggerMenuFile = 'L1Menu_Collisions2015_25nsStage1_v7_uGT.xml'
modifyL1TGlobalHackConditions_stage2 = stage2L1Trigger.makeProcessModifier(lambda p: _load(p, [
    "L1Trigger.L1TGlobal.GlobalParameters_cff",
    "L1Trigger.L1TGlobal.PrescalesVetos_cff",
    "L1Trigger.L1TGlobal.PrescalesVetosFract_cff",
#   "L1Trigger.L1TGlobal.TriggerMenu_cff"
]))
