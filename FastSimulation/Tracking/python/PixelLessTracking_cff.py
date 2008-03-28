import FWCore.ParameterSet.Config as cms

# Pixel Less seeding
from FastSimulation.Tracking.PixelLessSeedProducer_cff import *
import copy
from FastSimulation.Tracking.TrackCandidateProducer_cfi import *
# TrackCandidates
pixelLessGSTrackCandidates = copy.deepcopy(trackCandidateProducer)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
# reco::Tracks
pixelLessGSWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
# The sequence
pixelLessGSTracking = cms.Sequence(pixelLessGSSeeds*pixelLessGSTrackCandidates*pixelLessGSWithMaterialTracks)
pixelLessGSTrackCandidates.SeedProducer = cms.InputTag("pixelLessGSSeeds","PixelLess")
pixelLessGSWithMaterialTracks.src = 'pixelLessGSTrackCandidates'
pixelLessGSWithMaterialTracks.TTRHBuilder = 'WithoutRefit'
pixelLessGSWithMaterialTracks.Fitter = 'KFFittingSmoother'
pixelLessGSWithMaterialTracks.Propagator = 'PropagatorWithMaterial'

