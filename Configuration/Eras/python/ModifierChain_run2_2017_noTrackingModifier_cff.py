import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1

run2_2017_noTrackingModifier = Run2_2017.copyAndExclude([trackingPhase1])
