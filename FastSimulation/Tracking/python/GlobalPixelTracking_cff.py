import FWCore.ParameterSet.Config as cms

# Global Pixel seeding
from FastSimulation.Tracking.GlobalPixelSeedProducer_cff import *
import FastSimulation.Tracking.TrackCandidateProducer_cfi
# TrackCandidates
globalPixelTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
# reco::Tracks (possibly with invalid hits)
globalPixelWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
# The sequence
globalPixelTracking = cms.Sequence(globalPixelSeeds*globalPixelTrackCandidates*globalPixelWithMaterialTracks)
globalPixelTrackCandidates.SeedProducer = cms.InputTag("globalPixelSeeds","GlobalPixel")
globalPixelWithMaterialTracks.src = 'globalPixelTrackCandidates'
globalPixelWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
globalPixelWithMaterialTracks.Fitter = 'KFFittingSmoother'
globalPixelWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

