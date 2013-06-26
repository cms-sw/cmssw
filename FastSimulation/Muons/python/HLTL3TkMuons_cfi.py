import FWCore.ParameterSet.Config as cms

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
hltL3TkTracksFromL2 = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone(
    src = 'hltL3TrackCandidateFromL2',
    TTRHBuilder = 'WithoutRefit',
    Fitter = 'hltKFFittingSmoother',
    Propagator = 'PropagatorWithMaterial',
    beamSpot = 'offlineBeamSpot'
    )
