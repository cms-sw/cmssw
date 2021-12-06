import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitorClient.TrackingEffFromHitPatternClientConfig_cff import *
trackingEffFromHitPatternZeroBias = trackingEffFromHitPattern.clone(
    subDirs = (
        "Tracking/TrackParameters/generalTracks/HitEffFromHitPattern*",
        "Tracking/TrackParameters/highPurityTracks/pt_1/HitEffFromHitPattern*",
        "Tracking/TrackParameters/highPurityTracks/dzPV0p1/HitEffFromHitPattern*",
        "Tracking/TrackParameters/highPurityTracks/pt_1/HIP_OOT_noINpu/HitEffFromHitPattern*",
        "Tracking/TrackParameters/highPurityTracks/pt_1/HIP_noOOT_INpu/HitEffFromHitPattern*",
        "Tracking/TrackParameters/highPurityTracks/pt_1/noHIP_noOOT_INpu/HitEffFromHitPattern*",
        "Muons/Tracking/innerTrack/HitEffFromHitPattern*",
        "Muons/globalMuons/HitEffFromHitPattern*",
    )
)
