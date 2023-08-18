import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQM_cfg import *
DQMStore.collateHistograms =cms.untracked.bool(True)
from DQM.TrackingMonitorSource.TrackToTrackComparisonHists_cfi import TrackToTrackComparisonHists

trackSelector = cms.EDFilter('TrackSelector',
    src = cms.InputTag('generalTracks'),
    cut = cms.string("")
)
highPurityTracks = trackSelector.clone(
    cut = "quality('highPurity')"
)

hltMerged2highPurity = TrackToTrackComparisonHists.clone(
    monitoredTrack           = "hltMergedTracks",
    referenceTrack           = "highPurityTracks",
    monitoredBeamSpot        = "hltOnlineBeamSpot",
    referenceBeamSpot        = "offlineBeamSpot",
    topDirName               = "HLT/Tracking/ValidationWRTOffline/hltMergedWrtHighPurity",
    referencePrimaryVertices = "offlinePrimaryVertices",
    monitoredPrimaryVertices = "hltVerticesPFSelector"
)

hltMerged2highPurityPV = TrackToTrackComparisonHists.clone(
    dzWRTPvCut               = 0.1,
    monitoredTrack           = "hltMergedTracks",
    referenceTrack           = "highPurityTracks",
    monitoredBeamSpot        = "hltOnlineBeamSpot",
    referenceBeamSpot        = "offlineBeamSpot",
    topDirName               = "HLT/Tracking/ValidationWRTOffline/hltMergedWrtHighPurityPV",
    referencePrimaryVertices = "offlinePrimaryVertices",
    monitoredPrimaryVertices = "hltVerticesPFSelector"
)
hltToOfflineTrackValidatorSequence = cms.Sequence(
    cms.ignore(highPurityTracks)
    + hltMerged2highPurity
    + hltMerged2highPurityPV
)
