import FWCore.ParameterSet.Config as cms

#
# Pixel tracks
from RecoPixelVertexing.PixelLowPtUtilities.PixelLowPtTracksWithZPos_cff import *
import copy
from RecoPixelVertexing.PixelLowPtUtilities.PixelLowPtTracksWithZPos_cfi import *
# Pixel triplet tracks
pixel3ProtoTracks = copy.deepcopy(pixelLowPtTracksWithZPos)
# Pixel triplet vertices
from RecoPixelVertexing.PixelVertexFinding.PixelVertexes_cff import *
import copy
from RecoPixelVertexing.PixelLowPtUtilities.PixelLowPtTracksWithZPos_cfi import *
# Pixel triplet tracks with vertices
pixel3PrimTracks = copy.deepcopy(pixelLowPtTracksWithZPos)
# Hit remover
from RecoPixelVertexing.PixelLowPtUtilities.SiPixelFreeRecHits_cff import *
import copy
from RecoPixelVertexing.PixelLowPtUtilities.SiPixelFreeRecHits_cfi import *
# Pixel triplet tracks, secondaries  
pixelFreeSecoHits = copy.deepcopy(pixelFreeHits)
import copy
from RecoPixelVertexing.PixelLowPtUtilities.PixelLowPtTracksWithZPos_cfi import *
pixelSecoTracks = copy.deepcopy(pixelLowPtTracksWithZPos)
pixel3ProtoTracks.passLabel = 'Pixel triplet tracks for vertexing'
pixelVertices.TrackCollection = 'pixel3ProtoTracks'
pixelVertices.ZSeparation = 0.3
pixelVertices.NTrkMin = 3
pixelVertices.PtMin = 0.15
pixel3PrimTracks.passLabel = 'Pixel triplet primary tracks with vertex constraint'
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.useFoundVertices = True
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.originHalfLength = 0.2
pixelFreeSecoHits.removeHitsList = ['pixel3PrimTracks']
pixelSecoTracks.passLabel = 'Pixel triplet secondary tracks'
pixelSecoTracks.RegionFactoryPSet.RegionPSet.originRadius = 3.5
pixelSecoTracks.OrderedHitsFactoryPSet.SeedingLayers = 'PixelLayerTripletsModified'

