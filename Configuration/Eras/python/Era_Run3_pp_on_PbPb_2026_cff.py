import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2026_cff import Run3_2026
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.ProcessModifiers.rpdReco_cff import rpdReco
from Configuration.ProcessModifiers.hiEGReg_cff import hiEGReg
from Configuration.Eras.Modifier_dedx_lfit_cff import dedx_lfit
from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
from Configuration.Eras.Modifier_pp_on_PbPb_run3_2026_cff import pp_on_PbPb_run3_2026

Run3_pp_on_PbPb_2026 = cms.ModifierChain(Run3_2026, pp_on_AA, rpdReco, hiEGReg, dedx_lfit, pp_on_PbPb_run3, pp_on_PbPb_run3_2026)
