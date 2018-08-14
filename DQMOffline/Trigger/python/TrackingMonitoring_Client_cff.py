import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorClient.TrackingEffFromHitPatternClientConfig_cff import trackingEffFromHitPattern

trackingEffFromHitPatternHLT = trackingEffFromHitPattern.clone()
trackingEffFromHitPatternHLT.subDirs = cms.untracked.vstring(
   "HLT/Tracking/pixelTracks/HitEffFromHitPattern*",
   "HLT/Tracking/iter0HP/HitEffFromHitPattern*",
   "HLT/Tracking/iter2Merged/HitEffFromHitPattern*",
   "HLT/Tracking/tracks/HitEffFromHitPattern*"
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
   "HLT/EG/Tracking/iter0HP/HitEffFromHitPattern*",
   "HLT/EG/Tracking/iter2Merged/HitEffFromHitPattern*"
)

# Sequence
trackingForElectronsMonitorClientHLT = cms.Sequence(
    trackingForElectronsEffFromHitPatternHLT
)
