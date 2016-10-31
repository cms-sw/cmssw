import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.Eras.ModifierChain_run2_2017_core_cff import run2_2017_core
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
from Configuration.Eras.Modifier_run2_HE_2017_cff import run2_HE_2017
from Configuration.Eras.Modifier_run2_HF_2017_cff import run2_HF_2017
from Configuration.Eras.Modifier_run2_HCAL_2017_cff import run2_HCAL_2017

Run2_2017 = cms.ModifierChain(run2_2017_core, trackingPhase1)

