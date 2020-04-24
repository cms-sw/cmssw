import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_50ns_cff import Run2_50ns
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016

Run2_50ns_HIPM = cms.ModifierChain(Run2_50ns, tracker_apv_vfp30_2016)
