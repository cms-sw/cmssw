import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
from Configuration.Eras.Modifier_highBetaStar_2018_cff import highBetaStar_2018
from Configuration.Eras.Modifier_run3_upc_cff import run3_upc

Run3_UPC = cms.ModifierChain(Run3, egamma_lowPt_exclusive, highBetaStar_2018, run3_upc)
