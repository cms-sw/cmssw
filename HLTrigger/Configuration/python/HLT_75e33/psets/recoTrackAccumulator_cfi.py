import FWCore.ParameterSet.Config as cms

recoTrackAccumulator = cms.PSet(
    accumulatorType = cms.string('RecoTrackAccumulator'),
    makeDigiSimLinks = cms.untracked.bool(False),
    outputLabel = cms.string('generalTracks'),
    pileUpTracks = cms.InputTag("generalTracksBeforeMixing"),
    signalTracks = cms.InputTag("generalTracksBeforeMixing")
)