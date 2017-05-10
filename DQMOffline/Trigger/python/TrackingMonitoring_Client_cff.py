import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorClient.TrackingEffFromHitPatternClientConfig_cff import trackingEffFromHitPattern

trackingEffFromHitPatternHLT = trackingEffFromHitPattern.clone()
trackingEffFromHitPatternHLT.subDirs = cms.untracked.vstring(
   "HLT/Tracking/pixelTracks/HitEffFromHitPattern",
   "HLT/Tracking/iter0HP/HitEffFromHitPattern",
   "HLT/Tracking/iter2Merged/HitEffFromHitPattern"
)

# Sequence
trackingMonitorClientHLT = cms.Sequence(
    trackingEffFromHitPatternHLT
)
