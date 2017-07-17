import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
from Configuration.Eras.Modifier_trackingLowPU_cff import trackingLowPU

Run2_2016_trackingLowPU = cms.ModifierChain(Run2_2016, trackingLowPU)

