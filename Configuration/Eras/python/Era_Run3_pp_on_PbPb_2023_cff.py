import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_pp_on_PbPb_cff import Run3_pp_on_PbPb
from Configuration.Eras.Modifier_run3_egamma_2023_cff import run3_egamma_2023

Run3_pp_on_PbPb_2023 = cms.ModifierChain(Run3_pp_on_PbPb, run3_egamma_2023)
