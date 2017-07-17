import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.Eras.Modifier_run2_50ns_specific_cff import run2_50ns_specific

Run2_50ns = cms.ModifierChain(run2_common, run2_50ns_specific)

