import FWCore.ParameterSet.Config as cms

import DQM.TrackingMonitorClient.TrackingEffFromHitPatternClientConfig_cff
trackingEffFromHitPatternZeroBias = DQM.TrackingMonitorClient.TrackingEffFromHitPatternClientConfig_cff.trackingEffFromHitPattern.clone(
    subDirs = cms.untracked.vstring(
        "Tracking/TrackParameters/generalTracks/HitEffFromHitPattern*",
        "Tracking/TrackParameters/highPurityTracks/pt_1/HitEffFromHitPattern*",
        "Tracking/TrackParameters/highPurityTracks/dzPV0p1/HitEffFromHitPattern*",
        "Tracking/TrackParameters/highPurityTracks/pt_1/HIP_OOT_noINpu/HitEffFromHitPattern*",
        "Tracking/TrackParameters/highPurityTracks/pt_1/HIP_noOOT_INpu/HitEffFromHitPattern*",
        "Tracking/TrackParameters/highPurityTracks/pt_1/noHIP_noOOT_INpu/HitEffFromHitPattern*",
        "Muons/Tracking/innerTrack/HitEffFromHitPattern*",
        "Muons/globalMuons/HitEffFromHitPattern*"
    )
)
