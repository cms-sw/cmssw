import FWCore.ParameterSet.Config as cms

trackAccumulator = cms.PSet(

    GeneralTrackInputSignal = cms.InputTag("generalTracksBeforeMixing"),
    GeneralTrackInputPileup = cms.InputTag("generalTracks"),
    GeneralTrackOutput = cms.string("generalTracks"),

    #GeneralTrackExtraInputSignal = cms.InputTag("generalTracksBeforeMixing"),
    #GeneralTrackExtraInputPileup = cms.InputTag("generalTracks"),
    GeneralTrackExtraOutput = cms.string("generalTracks"),

    accumulatorType = cms.string("RecoTrackAccumulator"),
    makeDigiSimLinks = cms.untracked.bool(False)
    )
