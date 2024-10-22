import FWCore.ParameterSet.Config as cms

# tracking monitor
from DQMOffline.Trigger.TrackingMonitoring_Client_cff import *

trackingEffFromHitPatternHLT = trackingEffFromHitPattern.clone(
subDirs = (
   "HLT/Tracking/pixelTracks/HitEffFromHitPattern*",
   "HLT/Tracking/iter2Merged/HitEffFromHitPattern*",
   "HLT/Tracking/tracks/HitEffFromHitPattern*"
)
)
# Sequence
trackingMonitorClientHLT = cms.Sequence(
    trackingEffFromHitPatternHLT
)

# EGM tracking
trackingForElectronsEffFromHitPatternHLT = trackingEffFromHitPattern.clone(
subDirs = (
   "HLT/EGM/Tracking/GSF/HitEffFromHitPattern*",
   "HLT/EGM/Tracking/pixelTracks/HitEffFromHitPattern*",
   "HLT/EGM/Tracking/iter2Merged/HitEffFromHitPattern*"
)
)
# Sequence
trackingForElectronsMonitorClientHLT = cms.Sequence(
    trackingForElectronsEffFromHitPatternHLT
)

def _modifyForRun3Default(efffromhitpattern):
    efffromhitpattern.subDirs = ["HLT/Tracking/pixelTracks/HitEffFromHitPattern*", "HLT/Tracking/tracks/HitEffFromHitPattern*", "HLT/Tracking/doubletRecoveryTracks/HitEffFromHitPattern*"] #, "HLT/Tracking/iter0HP/HitEffFromHitPattern*"

def _modifyForRun3EGM(efffromhitpattern):
    efffromhitpattern.subDirs = ["HLT/EGM/Tracking/GSF/HitEffFromHitPattern*"]

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(trackingEffFromHitPatternHLT, _modifyForRun3Default)
run3_common.toModify(trackingForElectronsEffFromHitPatternHLT, _modifyForRun3EGM)
