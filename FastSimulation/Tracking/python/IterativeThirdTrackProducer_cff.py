import FWCore.ParameterSet.Config as cms

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeThirdTracksWithPairs = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeThirdTracks = cms.Sequence(iterativeThirdTracksWithPairs)
iterativeThirdTracksWithPairs.src = 'iterativeThirdTrackCandidatesWithPairs'
iterativeThirdTracksWithPairs.TTRHBuilder = 'WithoutRefit'
iterativeThirdTracksWithPairs.Fitter = 'KFFittingSmootherWithOutliersRejectionAndRK'
iterativeThirdTracksWithPairs.Propagator = 'PropagatorWithMaterial'


