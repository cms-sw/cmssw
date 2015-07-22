import FWCore.ParameterSet.Config as cms

# Pixel Less seeding
from FastSimulation.Tracking.PixelLessSeedProducer_cff import *
import FastSimulation.Tracking.TrackCandidateProducer_cfi
# TrackCandidates
pixelLessTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
# reco::Tracks
pixelLessWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
# The sequence
pixelLessTracking = cms.Sequence(pixelLessSeeds*pixelLessTrackCandidates*pixelLessWithMaterialTracks)
pixelLessTrackCandidates.src = cms.InputTag("pixelLessSeeds")
pixelLessWithMaterialTracks.src = 'pixelLessTrackCandidates'
pixelLessWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
pixelLessWithMaterialTracks.Fitter = 'KFFittingSmootherWithOutlierRejection'
pixelLessWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

