import FWCore.ParameterSet.Config as cms

############################
# Pixel-3 proto tracks
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi
pixel3ProtoTracks = RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi.allPixelTracks.clone()
pixel3ProtoTracks.passLabel = 'Pixel triplet tracks for vertexing'
pixel3ProtoTracks.RegionFactoryPSet.ptMin = cms.double(0.2)
pixel3ProtoTracks.FilterPSet.ptMin        = cms.double(0.2)

############################
# Pixel vertexing
#from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cfi import *
#pixel3Vertices = RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cfi.pixelVertices.clone()
#pixel3Vertices.TrackCollection = 'pixel3ProtoTracks'
#pixel3Vertices.UseError    = True
#pixel3Vertices.WtAverage   = True
#pixel3Vertices.ZOffset     = 5.
#pixel3Vertices.ZSeparation = 0.3
#pixel3Vertices.NTrkMin     = 3
#pixel3Vertices.PtMin       = 0.150
 
import UserCode.FerencSiklerVertexing.NewVertexProducer_cfi
pixel3Vertices = UserCode.FerencSiklerVertexing.NewVertexProducer_cfi.newVertices.clone()
pixel3Vertices.TrackCollection = 'pixel3ProtoTracks'
 
############################
# Pixel-3 primary tracks
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi
pixel3PrimTracks  = RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi.allPixelTracks.clone()
pixel3PrimTracks.passLabel  = 'Pixel triplet tracks with vertex constraint'
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.useFoundVertices = cms.bool(True)

############################
# Primary seeds
import RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi
primSeeds = RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone()
primSeeds.InputCollection = 'pixel3PrimTracks'

############################
# Primary track candidates
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
primTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
primTrackCandidates.TrajectoryCleaner    = 'TrajectoryCleanerBySharedSeeds'
primTrackCandidates.src                  = 'primSeeds'
primTrackCandidates.RedundantSeedCleaner = 'none'
primTrackCandidates.useHitsSplitting          = cms.bool(False)
primTrackCandidates.doSeedingRegionRebuilding = cms.bool(False)

############################
# Global primary tracks
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
globalPrimTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
globalPrimTracks.src               = 'primTrackCandidates'
globalPrimTracks.TrajectoryInEvent = cms.bool(True)
