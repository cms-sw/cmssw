import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.Eras.Modifier_run2_25ns_specific_cff import run2_25ns_specific
from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger

Run2_25ns = cms.ModifierChain(run2_common, run2_25ns_specific, stage1L1Trigger)

