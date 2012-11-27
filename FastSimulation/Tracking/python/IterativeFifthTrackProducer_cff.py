import FWCore.ParameterSet.Config as cms

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeFifthTracksWithPairs = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeFifthTracks = cms.Sequence(iterativeFifthTracksWithPairs)
iterativeFifthTracksWithPairs.src = 'iterativeFifthTrackCandidatesWithPairs'
iterativeFifthTracksWithPairs.TTRHBuilder = 'WithoutRefit'
iterativeFifthTracksWithPairs.Fitter = 'KFFittingSmootherFifth'
iterativeFifthTracksWithPairs.Propagator = 'PropagatorWithMaterial'


