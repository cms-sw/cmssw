import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorClient.TrackingEffFromHitPatternClientConfig_cff import trackingEffFromHitPattern

PAtrackingEffFromHitPatternHLT = trackingEffFromHitPattern.clone()
PAtrackingEffFromHitPatternHLT.subDirs = cms.untracked.vstring(
   "HLT/TrackingPA/pixelTracks/HitEffFromHitPattern*",
   "HLT/TrackingPA/pixelTracksForHighMult/HitEffFromHitPattern*",
   "HLT/TrackingPA/iter0/HitEffFromHitPattern*",
   "HLT/TrackingPA/iter1/HitEffFromHitPattern*",
   "HLT/TrackingPA/iter2/HitEffFromHitPattern*",
   "HLT/TrackingPA/iter3/HitEffFromHitPattern*",
   "HLT/TrackingPA/iter4/HitEffFromHitPattern*",
   "HLT/TrackingPA/iter5/HitEffFromHitPattern*",
   "HLT/TrackingPA/iter6/HitEffFromHitPattern*",
   "HLT/TrackingPA/iter7/HitEffFromHitPattern*",
   "HLT/TrackingPA/iterMerged/HitEffFromHitPattern*",
)

# Sequence
PAtrackingMonitorClientHLT = cms.Sequence(
    PAtrackingEffFromHitPatternHLT
)
