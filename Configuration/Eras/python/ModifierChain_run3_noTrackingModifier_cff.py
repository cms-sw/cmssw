import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1

run3_noTrackingModifier = Run3.copyAndExclude([trackingPhase1])
