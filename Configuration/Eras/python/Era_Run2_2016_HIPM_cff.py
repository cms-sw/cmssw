import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016

Run2_2016_HIPM = cms.ModifierChain(Run2_2016, tracker_apv_vfp30_2016)

