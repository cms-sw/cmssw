import FWCore.ParameterSet.Config as cms

# tracking monitor
from DQMOffline.Trigger.TrackingMonitoring_Client_cff import *

trackingEffFromHitPatternHLT = trackingEffFromHitPattern.clone()
trackingEffFromHitPatternHLT.subDirs = cms.untracked.vstring(
   "HLT/Tracking/pixelTracks/HitEffFromHitPattern*",
   "HLT/Tracking/iter2Merged/HitEffFromHitPattern*"
)

# Sequence
trackingMonitorClientHLT = cms.Sequence(
    trackingEffFromHitPatternHLT
)

# EGM tracking
trackingForElectronsEffFromHitPatternHLT = trackingEffFromHitPattern.clone()
trackingForElectronsEffFromHitPatternHLT.subDirs = cms.untracked.vstring(
   "HLT/EG/Tracking/GSF/HitEffFromHitPattern*",
   "HLT/EG/Tracking/pixelTracks/HitEffFromHitPattern*",
   "HLT/EG/Tracking/iter2Merged/HitEffFromHitPattern*"
)

# Sequence
trackingForElectronsMonitorClientHLT = cms.Sequence(
    trackingForElectronsEffFromHitPatternHLT
)
