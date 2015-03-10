#######################
# pset that configures the track accumulator, used in the MixingModules to mix reconstructed tracks
# author: Lukas Vanelderen
# date:   Jan 21 2015
#######################

import FWCore.ParameterSet.Config as cms

recoTrackAccumulator = cms.PSet(
    InputSignal = cms.InputTag("generalTracksBeforeMixing"),
    InputPileUp = cms.InputTag("generalTracksBeforeMixing"),

    GeneralTrackOutput = cms.string("generalTracks"),
    GeneralTrackExtraOutput = cms.string("generalTracks"),
    HitOutput = cms.string("generalTracks"),

    accumulatorType = cms.string("RecoTrackAccumulator"),
    makeDigiSimLinks = cms.untracked.bool(False)
    )
