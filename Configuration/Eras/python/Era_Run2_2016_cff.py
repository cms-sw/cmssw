import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.Eras.Modifier_run2_25ns_specific_cff import run2_25ns_specific
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016

Run2_2016 = cms.ModifierChain(run2_common, run2_25ns_specific, stage2L1Trigger, ctpps_2016)

