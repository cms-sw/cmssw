import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQM_cfg import *
DQMStore.collateHistograms =cms.untracked.bool(True)
from DQM.TrackingMonitorSource.TrackToTrackValidator_cfi import *

trackSelector = cms.EDFilter('TrackSelector',
    src = cms.InputTag('generalTracks'),
    cut = cms.string("")
)
highPurityTracks = trackSelector.clone()
highPurityTracks.cut = cms.string("quality('highPurity')")

hltMerged2hltMerged = trackToTrackValidator.clone()
hltMerged2hltMerged.monitoredTrack      = cms.InputTag("hltMergedTracks")
hltMerged2hltMerged.referenceTrack      = cms.InputTag("hltMergedTracks")
hltMerged2hltMerged.monitoredBeamSpot   = cms.InputTag("hltOnlineBeamSpot")
hltMerged2hltMerged.referenceBeamSpot   = cms.InputTag("hltOnlineBeamSpot")
hltMerged2hltMerged.topDirName    = cms.string("HLT/Tracking/ValidationWRTOffline/sanity")
hltMerged2hltMerged.referencePrimaryVertices = cms.InputTag("hltVerticesPFSelector")
hltMerged2hltMerged.monitoredPrimaryVertices = cms.InputTag("hltVerticesPFSelector")

hltMerged2generalTracks = trackToTrackValidator.clone()
hltMerged2generalTracks.monitoredTrack      = cms.InputTag("hltMergedTracks")
hltMerged2generalTracks.referenceTrack      = cms.InputTag("generalTracks")
hltMerged2generalTracks.monitoredBeamSpot   = cms.InputTag("hltOnlineBeamSpot")
hltMerged2generalTracks.referenceBeamSpot   = cms.InputTag("offlineBeamSpot")
hltMerged2generalTracks.topDirName    = cms.string("HLT/Tracking/ValidationWRTOffline/hltMerged")
hltMerged2generalTracks.referencePrimaryVertices = cms.InputTag("offlinePrimaryVertices")
hltMerged2generalTracks.monitoredPrimaryVertices = cms.InputTag("hltVerticesPFSelector")

hltMerged2highPurity = trackToTrackValidator.clone()
hltMerged2highPurity.monitoredTrack      = cms.InputTag("hltMergedTracks")
hltMerged2highPurity.referenceTrack      = cms.InputTag("highPurityTracks")
hltMerged2highPurity.monitoredBeamSpot   = cms.InputTag("hltOnlineBeamSpot")
hltMerged2highPurity.referenceBeamSpot   = cms.InputTag("offlineBeamSpot")
hltMerged2highPurity.topDirName    = cms.string("HLT/Tracking/ValidationWRTOffline/hltMergedWrtHighPurity")
hltMerged2highPurity.referencePrimaryVertices = cms.InputTag("offlinePrimaryVertices")
hltMerged2highPurity.monitoredPrimaryVertices = cms.InputTag("hltVerticesPFSelector")


hltToOfflineTrackValidatorSequence = cms.Sequence(
    cms.ignore(highPurityTracks)+
    hltMerged2hltMerged+
    hltMerged2generalTracks+
    hltMerged2highPurity
)
