import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cfi import *
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *

import RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi
pixel3ProtoTracks = RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi.allPixelTracks.clone()
pixel3ProtoTracks.passLabel = 'Pixel triplet tracks for vertexing'

import RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi
pixel3PrimTracks = RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi.allPixelTracks.clone()
pixel3PrimTracks.passLabel = 'Pixel triplet tracks with vertex constraint'
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.useFoundVertices = True

import RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi
primSeeds = RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone()
primSeeds.tripletList = ['pixel3PrimTracks']

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
primTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
primTrackCandidates.TrajectoryCleaner    = 'TrajectoryCleanerBySharedSeeds'
primTrackCandidates.SeedProducer         = 'primSeeds'
primTrackCandidates.RedundantSeedCleaner = 'none'

primTrackCandidates.useHitsSplitting          = cms.bool(False)
primTrackCandidates.doSeedingRegionRebuilding = cms.bool(False)

import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
globalPrimTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
globalPrimTracks.src               = 'primTrackCandidates'
globalPrimTracks.TrajectoryInEvent = True

pixelVertices.TrackCollection = 'pixel3ProtoTracks'
pixelVertices.UseError    = True
pixelVertices.WtAverage   = True
pixelVertices.ZOffset     = 5.
pixelVertices.ZSeparation = 0.3
pixelVertices.NTrkMin     = 3
pixelVertices.PtMin       = 0.150

