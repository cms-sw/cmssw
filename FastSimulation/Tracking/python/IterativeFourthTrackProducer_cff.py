import FWCore.ParameterSet.Config as cms

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
iterativeFourthTracksWithPairs = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
iterativeFourthTracks = cms.Sequence(iterativeFourthTracksWithPairs)
iterativeFourthTracksWithPairs.src = 'iterativeFourthTrackCandidatesWithPairs'
iterativeFourthTracksWithPairs.TTRHBuilder = 'WithoutRefit'
##iterativeFourthTracksWithPairs.Fitter = 'KFFittingSmootherWithOutlierRejection'
iterativeFourthTracksWithPairs.Fitter = 'KFFittingSmootherFourth'
iterativeFourthTracksWithPairs.Propagator = 'PropagatorWithMaterial'


