import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2026_cff import Run3_2026
from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
from Configuration.ProcessModifiers.rpdReco_cff import rpdReco
from Configuration.Eras.Modifier_highBetaStar_cff import highBetaStar
from Configuration.Eras.Modifier_dedx_lfit_cff import dedx_lfit
from Configuration.Eras.Modifier_run3_upc_cff import run3_upc
from Configuration.Eras.Modifier_run3_upc_2026_cff import run3_upc_2026

Run3_2026_UPC = cms.ModifierChain(Run3_2026, egamma_lowPt_exclusive, rpdReco, highBetaStar, dedx_lfit, run3_upc, run3_upc_2026)
