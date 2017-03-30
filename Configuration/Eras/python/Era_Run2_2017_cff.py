import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017
from Configuration.Eras.Modifier_run2_HE_2017_cff import run2_HE_2017
from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017

Run2_2017 = cms.ModifierChain(Run2_2016, phase1Pixel, run2_HF_2017, run2_HCAL_2017, run2_HE_2017, run2_HEPlan1_2017, trackingPhase1, run2_GEM_2017)

