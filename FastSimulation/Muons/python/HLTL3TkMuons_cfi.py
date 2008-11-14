import FWCore.ParameterSet.Config as cms

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
hltL3TkTracksFromL2 = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltL3TkTracksFromL2.src = 'hltL3TrackCandidateFromL2'
hltL3TkTracksFromL2.TTRHBuilder = 'WithoutRefit'
hltL3TkTracksFromL2.Fitter = 'hltKFFittingSmoother'
hltL3TkTracksFromL2.Propagator = 'PropagatorWithMaterial'
hltL3TkTracksFromL2.beamSpot = 'offlineBeamSpot'
