#
# hachConditions.py  Load ES Producers for any conditions not yet in GT...
#
# The intention is that this file should shrink with time as conditions are added to GT.
#

import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

#
# Legacy Trigger:  No Hacks Needed
#
#if not (eras.stage1L1Trigger.isChosen() or eras.stage2L1Trigger.isChosen()):
#    print "L1TCalorimeter conditions configured for Run1 (Legacy) trigger. "
# 

#
# Stage-1 Trigger
#
if eras.stage1L1Trigger.isChosen() and not eras.stage2L1Trigger.isChosen():
    print "L1TCalorimeter Conditions configured for Stage-1 (2015) trigger. "    
    # Switch between HI and PP calo configuration:
    if (eras.run2_HI_specific.isChosen()):
        from L1Trigger.L1TCalorimeter.caloConfigStage1HI_cfi import *
    else:
        from L1Trigger.L1TCalorimeter.caloConfigStage1PP_cfi import *
    # Override Calo Scales:
    from L1Trigger.L1TCalorimeter.caloScalesStage1_cff import *
    # CaloParams is in the DB for Stage-1

#
# Stage-2 Trigger
#
if eras.stage2L1Trigger.isChosen():
    print "L1TCalorimeter Conditions configured for Stage-2 (2016) trigger. "
    # from L1Trigger.L1TCalorimeter.simCaloStage2Layer1Digis_cfi import simCaloStage2Layer1Digis    
    from L1Trigger.L1TCalorimeter.caloStage2Params_2016_v2_1_cfi import *
    #
    # What about CaloConfig?  Related:  How will we switch PP/HH?
    #
