import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.cosmictrackSelector_cfi

cosmictrackfinderP5 = RecoTracker.FinalTrackSelectors.cosmictrackSelector_cfi.cosmictrackSelector.clone (
    src = "cosmictrackfinderCosmics"
)
