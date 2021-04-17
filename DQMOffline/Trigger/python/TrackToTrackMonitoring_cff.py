import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQM_cfg import *
DQMStore.collateHistograms =cms.untracked.bool(True)
from DQM.TrackingMonitorSource.trackToTrackComparisonHists_cfi import trackToTrackComparisonHists

trackSelector = cms.EDFilter('TrackSelector',
    src = cms.InputTag('generalTracks'),
    cut = cms.string("")
)
highPurityTracks = trackSelector.clone()
highPurityTracks.cut = cms.string("quality('highPurity')")


hltMerged2highPurity = trackToTrackComparisonHists.clone()
hltMerged2highPurity.monitoredTrack           = cms.InputTag("hltMergedTracks")
hltMerged2highPurity.referenceTrack           = cms.InputTag("highPurityTracks")
hltMerged2highPurity.monitoredBeamSpot        = cms.InputTag("hltOnlineBeamSpot")
hltMerged2highPurity.referenceBeamSpot        = cms.InputTag("offlineBeamSpot")
hltMerged2highPurity.topDirName               = cms.string("HLT/Tracking/ValidationWRTOffline/hltMergedWrtHighPurity")
hltMerged2highPurity.referencePrimaryVertices = cms.InputTag("offlinePrimaryVertices")
hltMerged2highPurity.monitoredPrimaryVertices = cms.InputTag("hltVerticesPFSelector")


hltMerged2highPurityPV = trackToTrackComparisonHists.clone()
hltMerged2highPurityPV.dzWRTPvCut               = cms.double(0.1)
hltMerged2highPurityPV.monitoredTrack           = cms.InputTag("hltMergedTracks")
hltMerged2highPurityPV.referenceTrack           = cms.InputTag("highPurityTracks")
hltMerged2highPurityPV.monitoredBeamSpot        = cms.InputTag("hltOnlineBeamSpot")
hltMerged2highPurityPV.referenceBeamSpot        = cms.InputTag("offlineBeamSpot")
hltMerged2highPurityPV.topDirName               = cms.string("HLT/Tracking/ValidationWRTOffline/hltMergedWrtHighPurityPV")
hltMerged2highPurityPV.referencePrimaryVertices = cms.InputTag("offlinePrimaryVertices")
hltMerged2highPurityPV.monitoredPrimaryVertices = cms.InputTag("hltVerticesPFSelector")

hltToOfflineTrackValidatorSequence = cms.Sequence(
    cms.ignore(highPurityTracks)
    + hltMerged2highPurity
    + hltMerged2highPurityPV
)
