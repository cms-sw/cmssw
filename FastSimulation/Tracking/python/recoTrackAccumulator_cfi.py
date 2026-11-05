#######################
# pset that configures the track accumulator, used in the MixingModules to mix reconstructed tracks
# author: Lukas Vanelderen
# date:   Jan 21 2015
#######################

import FWCore.ParameterSet.Config as cms

recoTrackAccumulator = cms.PSet(
    signalTracks = cms.InputTag("generalTracksBeforeMixing"),
    pileUpTracks = cms.InputTag("generalTracksBeforeMixing"),
    copyTrajectories = cms.bool(False),
    outputLabel = cms.string("generalTracks"),
    accumulatorType = cms.string("RecoTrackAccumulator"),
    makeDigiSimLinks = cms.untracked.bool(False)
)

from Configuration.Eras.Modifier_phase2_timing_layer_cff import phase2_timing_layer
phase2_timing_layer.toModify(recoTrackAccumulator,
    copyTrajectories = True
)
