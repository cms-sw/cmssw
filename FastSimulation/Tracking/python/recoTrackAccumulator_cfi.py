import FWCore.ParameterSet.Config as cms

trackAccumulator = cms.PSet(
    GeneralTrackInput = cms.InputTag("generalTracks"),
    GeneralTrackOutput = cms.string("generalTracks"),
    accumulatorType = cms.string("RecoTrackAccumulator"),
    makeDigiSimLinks = cms.untracked.bool(False)
    )
