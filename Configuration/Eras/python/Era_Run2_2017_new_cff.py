import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Modifier_phase1PixelNewFPix_cff import phase1PixelNewFPix
from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017

Run2_2017_new = cms.ModifierChain(Run2_2017, phase1PixelNewFPix, run2_HF_2017, run2_HCAL_2017)

