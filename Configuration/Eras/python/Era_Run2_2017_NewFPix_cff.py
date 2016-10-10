import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Modifier_phase1PixelNewFPix_cff import phase1PixelNewFPix

Run2_2017_NewFPix = cms.ModifierChain(Run2_2017, phase1PixelNewFPix)

