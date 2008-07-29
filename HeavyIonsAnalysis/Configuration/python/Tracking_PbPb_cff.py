import FWCore.ParameterSet.Config as cms


from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *

from RecoPixelVertexing.PixelLowPtUtilities.common_cff import *
from RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi
pixel3ProtoTracks = RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi.allPixelTracks.clone()
import RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi
pixel3PrimTracks = RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi.allPixelTracks.clone()
import RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi
primSeeds = RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
primTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
globalPrimTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
from RecoVZero.VZeroFinding.VZeros_cff import *
pixelVertices = cms.EDProducer("PixelVertexProducerMedian",
                               TrackCollection = cms.string("pixel3ProtoTracks"),
                               PtMin = cms.double(0.5)
                               )
firstStep = cms.Sequence(pixel3ProtoTracks*pixelVertices*pixel3PrimTracks*primSeeds*primTrackCandidates*globalPrimTracks)
heavyIonTracking = cms.Sequence(firstStep)

esprefRHProd = cms.ESPrefer("TkTransientTrackingRecHitBuilderESProducer","myBuilder")

pixel3ProtoTracks.passLabel = 'Pixel triplet tracks for vertexing'
pixel3ProtoTracks.RegionFactoryPSet.RegionPSet.ptMin = 0.5
pixel3ProtoTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.1
pixel3PrimTracks.passLabel = 'Pixel triplet tracks with vertex constraint'
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.useFoundVertices = True
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.VertexCollection = 'pixelVertices'
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.fixedError = 0.2
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.ptMin = 0.5
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.2
primSeeds.tripletList = ['pixel3PrimTracks']
primTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedSeeds'
primTrackCandidates.SeedProducer = 'primSeeds'
primTrackCandidates.RedundantSeedCleaner = 'none'
globalPrimTracks.src = 'primTrackCandidates'
globalPrimTracks.TrajectoryInEvent = True
# pixelVZeros.trackProducer = 'globalPrimTracks'

hiTrackingWithOfflineBeamSpot = cms.Sequence(offlineBeamSpot*trackerlocalreco*heavyIonTracking)

