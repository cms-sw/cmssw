import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.Eras.Modifier_run2_HI_specific_cff import run2_HI_specific
from Configuration.Eras.Modifier_stage2L1Trigger_2017_cff import stage2L1Trigger_2017

Run2_HI = cms.ModifierChain(run2_common, run2_HI_specific, stage2L1Trigger_2017)

