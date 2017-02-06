#
# hachConditions.py  Load ES Producers for any conditions not yet in GT...
#
# The intention is that this file should shrink with time as conditions are added to GT.
#

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run2_HI_specific_cff import run2_HI_specific
#from Configuration.Eras.Era_Run2_2016_pA_cff import Run2_2016_pA
from Configuration.Eras.Modifier_pA_2016_cff import pA_2016


#
# Legacy Trigger:  No Hacks Needed
#
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
#if not (stage1L1Trigger.isChosen() or stage2L1Trigger.isChosen()):
#    print "L1TCalorimeter conditions configured for Run1 (Legacy) trigger. "
# 

#
# Stage-1 Trigger
#
if stage1L1Trigger.isChosen() and not stage2L1Trigger.isChosen():
    print "L1TCalorimeter Conditions configured for Stage-1 (2015) trigger. "    
    # Switch between HI and PP calo configuration:
    if (run2_HI_specific.isChosen()):
        from L1Trigger.L1TCalorimeter.caloConfigStage1HI_cfi import *
    else:
        from L1Trigger.L1TCalorimeter.caloConfigStage1PP_cfi import *
    # Override Calo Scales:
    from L1Trigger.L1TCalorimeter.caloScalesStage1_cff import *
    # CaloParams is in the DB for Stage-1

#
# Stage-2 Trigger
#
if (stage2L1Trigger.isChosen() or phase2_common.isChosen()):
    if pA_2016.isChosen():
        print "L1TCalorimeter Conditions configured for Stage-2 (2016 pA) trigger. "
        from L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_3_1_HI_cfi import *    
    else:
        print "L1TCalorimeter Conditions configured for Stage-2 (2016) trigger. "
        from L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_3_1_cfi import *    
    
    # What about CaloConfig?  Related:  How will we switch PP/HH?
    #
