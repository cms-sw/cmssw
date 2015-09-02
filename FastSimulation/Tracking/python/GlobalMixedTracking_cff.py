import FWCore.ParameterSet.Config as cms

# Global Mixed seeding
from FastSimulation.Tracking.GlobalMixedSeedProducer_cff import *
import FastSimulation.Tracking.TrackCandidateProducer_cfi
# TrackCandidates
globalMixedTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
# reco::Tracks
globalMixedWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()

# The sequence
ctfTracking = cms.Sequence(globalMixedSeeds*globalMixedTrackCandidates*globalMixedWithMaterialTracks*ctfWithMaterialTracks)
globalMixedTrackCandidates.src = cms.InputTag("globalMixedSeeds")
globalMixedTrackCandidates.TrackProducers = ['globalPixelWithMaterialTracks']
globalMixedWithMaterialTracks.src = 'globalMixedTrackCandidates'
globalMixedWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
globalMixedWithMaterialTracks.Fitter = 'KFFittingSmootherWithOutlierRejection'
globalMixedWithMaterialTracks.Propagator = 'PropagatorWithMaterial'
