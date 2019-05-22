#
# hachConditions.py  Load ES Producers for any conditions not yet in GT...
#
# The intention is that this file should shrink with time as conditions are added to GT.
#

import FWCore.ParameterSet.Config as cms
import sys
from Configuration.Eras.Modifier_run2_HI_specific_cff import run2_HI_specific
#from Configuration.Eras.Era_Run2_2016_pA_cff import Run2_2016_pA
from Configuration.Eras.Modifier_pA_2016_cff import pA_2016


#
# Legacy Trigger:  No Hacks Needed
#
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger

def _load(process, f):
    process.load(f)

#
# Stage-1 Trigger
#
# Switch between HI and PP calo configuration:
modifyL1TCalorimeterHackConditions_stage1HI = (stage1L1Trigger & ~stage2L1Trigger & run2_HI_specific).makeProcessModifier(lambda p: _load(p, "L1Trigger.L1TCalorimeter.caloConfigStage1HI_cfi"))
modifyL1TCalorimeterHackConditions_stage1PP = (stage1L1Trigger & ~stage2L1Trigger & ~run2_HI_specific).makeProcessModifier(lambda p: _load(p, "L1Trigger.L1TCalorimeter.caloConfigStage1PP_cfi"))
# Override Calo Scales:
modifyL1TCalorimeterHackConditions_stage1Common = (stage1L1Trigger & ~stage2L1Trigger).makeProcessModifier(lambda p: _load(p, "L1Trigger.L1TCalorimeter.caloScalesStage1_cff"))
# CaloParams is in the DB for Stage-1


#
# Stage-2 Trigger
#
modifyL1TCalorimeterHackConditions_stage2PA = (stage2L1Trigger & pA_2016).makeProcessModifier(lambda p: _load(p, "L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_3_1_HI_cfi"))
modifyL1TCalorimeterHackConditions_stage2PP = (stage2L1Trigger & ~pA_2016).makeProcessModifier(lambda p: _load(p, "L1Trigger.L1TCalorimeter.caloStage2Params_2017_v1_8_4_cfi"))
# What about CaloConfig?  Related:  How will we switch PP/HH?
#
