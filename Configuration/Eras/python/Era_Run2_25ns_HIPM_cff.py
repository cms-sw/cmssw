import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_25ns_cff import Run2_25ns
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016

Run2_25ns_HIPM = cms.ModifierChain(Run2_25ns, tracker_apv_vfp30_2016)
