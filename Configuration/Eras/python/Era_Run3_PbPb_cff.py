import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
from Configuration.Eras.Modifier_pbpb_run3_cff import pbpb_run3

Run3_PbPb = cms.ModifierChain(Run3, pp_on_AA_2018, pbpb_run3)
