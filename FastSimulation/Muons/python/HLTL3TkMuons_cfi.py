import FWCore.ParameterSet.Config as cms

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
hltL3TkMuons = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hltL3TkMuons.src = 'hltL3CandidateFromL2'
hltL3TkMuons.TTRHBuilder = 'WithoutRefit'
hltL3TkMuons.Fitter = 'KFFittingSmoother'
hltL3TkMuons.Propagator = 'PropagatorWithMaterial'
hltL3TkMuons.beamSpot = 'offlineBeamSpot'

