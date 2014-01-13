import FWCore.ParameterSet.Config as cms

trackAccumulator = cms.PSet(

    InputSignal = cms.InputTag("generalTracksBeforeMixing"),
    InputPileup = cms.InputTag("generalTracks"),

    GeneralTrackOutput = cms.string("generalTracks"),
    GeneralTrackExtraOutput = cms.string("generalTracks"),
    HitOutput = cms.string("generalTracks"),

    accumulatorType = cms.string("RecoTrackAccumulator"),
    makeDigiSimLinks = cms.untracked.bool(False)
    )
