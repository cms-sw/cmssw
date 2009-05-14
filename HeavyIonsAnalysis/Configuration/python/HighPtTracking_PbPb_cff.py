import FWCore.ParameterSet.Config as cms

from TrackingTools.Configuration.TrackingTools_cff import *
from TrackingTools.TransientTrack.TransientTrackBuilder_cfi import *
from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelTriplets_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
from RecoHI.HiTracking.PixelLowPtTracksWithZPos_cff import *
pixel3ProtoTracks = pixelLowPtTracksWithZPos.clone()
from RecoHI.HiTracking.PixelVertices_cfi import *
pixel3PrimTracks = pixelLowPtTracksWithZPos.clone()
from RecoHI.HiTracking.PixelTrackSeeds_cff import *
primSeeds = pixelTrackSeeds.clone()
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cff import *
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
ckfBaseTrajectoryFilterForMinBias = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
from RecoHI.HiTracking.TrajectoryCleanerBySharedSeeds_cfi import *
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
primTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
globalPrimTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
clusterShapeTrajectoryFilterESProducer = cms.ESProducer("ClusterShapeTrajectoryFilterESProducer",
    filterPset = cms.PSet(
        ComponentType = cms.string('clusterShapeTrajectoryFilter')
    ),
    ComponentName = cms.string('clusterShapeTrajectoryFilter')
)

minBiasTrajectoryFilterESProducer = cms.ESProducer("CompositeTrajectoryFilterESProducer",
    ComponentName = cms.string('MinBiasCkfTrajectoryFilter'),
    filterNames = cms.vstring('ckfBaseTrajectoryFilterForMinBias', 
        'clusterShapeTrajectoryFilter')
)

heavyIonTracking = cms.Sequence(pixel3ProtoTracks*pixelVertices*pixel3PrimTracks*primSeeds*primTrackCandidates*globalPrimTracks)
hiTrackingWithOfflineBeamSpot = cms.Sequence(offlineBeamSpot*trackerlocalreco*heavyIonTracking)
pixel3ProtoTracks.passLabel = 'Pixel triplet tracks for vertexing'
pixel3ProtoTracks.RegionFactoryPSet.RegionPSet.ptMin = 0.7
pixel3ProtoTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.1
pixel3ProtoTracks.RegionFactoryPSet.RegionPSet.originZPos = 2.0
pixelVertices.TrackCollection = 'pixel3ProtoTracks'
pixel3PrimTracks.passLabel = 'Pixel triplet primary tracks with vertex constraint'
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.ptMin = 1.5
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.2
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.originHalfLength = 0.2
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.useFoundVertices = True
primSeeds.tripletList = ['pixel3PrimTracks']
ckfBaseTrajectoryFilterForMinBias.ComponentName = 'ckfBaseTrajectoryFilterForMinBias'
ckfBaseTrajectoryFilterForMinBias.filterPset.minimumNumberOfHits = 6
ckfBaseTrajectoryFilterForMinBias.filterPset.minPt = 2.0
GroupedCkfTrajectoryBuilder.maxCand = 5
GroupedCkfTrajectoryBuilder.intermediateCleaning = False
GroupedCkfTrajectoryBuilder.alwaysUseInvalidHits = False
MaterialPropagator.Mass = 0.139
OppositeMaterialPropagator.Mass = 0.139
primTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedSeeds'
primTrackCandidates.TrajectoryBuilder = 'CkfTrajectoryBuilder'
primTrackCandidates.SeedProducer = 'primSeeds'
primTrackCandidates.RedundantSeedCleaner = 'none'
globalPrimTracks.src = 'primTrackCandidates'
globalPrimTracks.TrajectoryInEvent = True
globalPrimTracks.useHitsSplitting = True


