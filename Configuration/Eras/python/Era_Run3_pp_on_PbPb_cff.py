import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3

Run3_pp_on_PbPb = cms.ModifierChain(Run3, pp_on_PbPb_run3)
