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
# Stage-2 Trigger
#
def _load(process, fs):
    for f in fs:
        process.load(f)
modifyL1TMuonHackConditions_stage2 = stage2L1Trigger.makeProcessModifier(lambda p: _load(p, [
    "L1Trigger.L1TTwinMux.fakeTwinMuxParams_cff",
    "L1Trigger.L1TMuonBarrel.fakeBmtfParams_cff",
    "L1Trigger.L1TMuonOverlap.fakeOmtfParams_cff",
    "L1Trigger.L1TMuonEndCap.fakeEmtfParams_cff",
    "L1Trigger.L1TMuon.fakeGmtParams_cff"
]))
