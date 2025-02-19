import FWCore.ParameterSet.Config as cms

import RecoTracker.FinalTrackSelectors.cosmictrackSelector_cfi

rsWithMaterialTracksP5 = RecoTracker.FinalTrackSelectors.cosmictrackSelector_cfi.cosmictrackSelector.clone (
    src = "rsWithMaterialTracksCosmics"
    )
