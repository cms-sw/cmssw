import FWCore.ParameterSet.Config as cms
from RecoTracker.IterativeTracking.HighPtTripletStep_cff import HighPtTripletStepTaskSerialSync
from .TrackToTrackComparisonHists_cfi import TrackToTrackComparisonHists as _t2t

trackToTrackCPUSequence = cms.Sequence()

highPtTripletStepTrackToTrackSerialSync = _t2t.clone(
    requireValidHLTPaths = False,
    monitoredTrack = "highPtTripletStepTracks",
    referenceTrack = "highPtTripletStepTracksSerialSync",

    monitoredBeamSpot = "offlineBeamSpot",
    monitoredPrimaryVertices = "offlinePrimaryVertices",
    topDirName = "Tracking/TrackBuilding/ValidationWRTSerialSync/highPtTripletStep"
)
_trackToTrackCPUTask_trackingLST = cms.Sequence(HighPtTripletStepTaskSerialSync)
_trackToTrackCPUTask_trackingLST += highPtTripletStepTrackToTrackSerialSync

from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
from Configuration.ProcessModifiers.alpakaValidationLST_cff import alpakaValidationLST
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
(trackingPhase2PU140 & alpakaValidationLST & trackingLST).toReplaceWith(trackToTrackCPUSequence, _trackToTrackCPUTask_trackingLST)
