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
from Configuration.StandardSequences.Eras import eras

#
# Legacy Trigger:  No Hacks Needed
#
#if not (eras.stage1L1Trigger.isChosen() or eras.stage2L1Trigger.isChosen()):
#    print "L1TGlobal conditions configured for Run1 (Legacy) trigger. "
# 

#
# Stage-1 Trigger:  No Hacks Needed
#
#if eras.stage1L1Trigger.isChosen() and not eras.stage2L1Trigger.isChosen():
#    print "L1TGlobal Conditions configured for Stage-1 (2015) trigger. "    

#
# Stage-2 Trigger
#
if eras.stage2L1Trigger.isChosen():
    print "L1TGlobal Conditions configured for Stage-2 (2016) trigger. "

    # During transition to new uGT software for handling correlation conditions, we'll fill both version
    # of the stable parameters in the (to be deprecated) and (new) version of the stable conditions:
    # to be deprecated, but save to leave for some time:
    from L1Trigger.L1TGlobal.StableParameters_cff import *
    # the new version:
    from L1Trigger.L1TGlobal.GlobalParameters_cff import *
    # The following ES producer does nothing at the moment except write an empty format, it is here only to exercise this condition
    # in our tests.
    # Even when filled, these conditions will only be requested if re-emulating prescales, vetos, and
    # so HLT should *never* have to add such a dummy producer to their configs:
    from L1Trigger.L1TGlobal.PrescalesVetos_cff import *
#   L1Menu now taken from the Global Tag:
#   from L1Trigger.L1TGlobal.TriggerMenu_cff import *
#   TriggerMenu.L1TriggerMenuFile = cms.string('L1Menu_Collisions2015_25nsStage1_v7_uGT.xml')
