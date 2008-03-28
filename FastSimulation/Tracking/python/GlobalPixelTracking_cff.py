import FWCore.ParameterSet.Config as cms

# Global Pixel seeding
from FastSimulation.Tracking.GlobalPixelSeedProducer_cff import *
import copy
from FastSimulation.Tracking.TrackCandidateProducer_cfi import *
# TrackCandidates
globalPixelGSTrackCandidates = copy.deepcopy(trackCandidateProducer)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
# reco::Tracks (possibly with invalid hits)
globalPixelGSWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
# The sequence
globalPixelGSTracking = cms.Sequence(globalPixelGSSeeds*globalPixelGSTrackCandidates*globalPixelGSWithMaterialTracks)
globalPixelGSTrackCandidates.SeedProducer = cms.InputTag("globalPixelGSSeeds","GlobalPixel")
globalPixelGSWithMaterialTracks.src = 'globalPixelGSTrackCandidates'
globalPixelGSWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
globalPixelGSWithMaterialTracks.Fitter = 'KFFittingSmoother'
globalPixelGSWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

