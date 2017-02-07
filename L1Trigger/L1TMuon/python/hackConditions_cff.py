#
# hackConditions.py  Load ES Producers for any conditions not yet in GT...
#
# The intention is that this file should shrink with time as conditions are added to GT.
#

import FWCore.ParameterSet.Config as cms

#
# Legacy Trigger:  No Hacks Needed
#
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
#if not (stage1L1Trigger.isChosen() or stage2L1Trigger.isChosen()):
#    print "L1TMuon conditions configured for Run1 (Legacy) trigger. "
# 

#
# Stage-1 Trigger:  No Hacks Needed
#
#if stage1L1Trigger.isChosen() and not stage2L1Trigger.isChosen():
#    print "L1TMuon Conditions configured for Stage-1 (2015) trigger. "    

#
# Stage-2 Trigger
#
if (stage2L1Trigger.isChosen() or phase2_common.isChosen()):
    print "L1TMuon Conditions configured for Stage-2 (2016) trigger. "
    from L1Trigger.L1TMuonBarrel.fakeBmtfParams_cff import *
    from L1Trigger.L1TMuonOverlap.fakeOmtfParams_cff import *
    from L1Trigger.L1TMuonEndCap.fakeEmtfParams_cff import *
    from L1Trigger.L1TMuon.fakeGmtParams_cff import *
