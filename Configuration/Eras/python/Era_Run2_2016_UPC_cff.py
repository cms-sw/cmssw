import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
from Configuration.Eras.Modifier_highBetaStar_cff import highBetaStar
from Configuration.Eras.Modifier_run3_upc_cff import run3_upc

Run2_2016_UPC = cms.ModifierChain(Run2_2016, egamma_lowPt_exclusive, highBetaStar, run3_upc)
