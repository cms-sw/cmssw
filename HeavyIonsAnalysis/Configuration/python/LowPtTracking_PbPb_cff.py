import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *
from RecoTracker.TkSeedingLayers.TTRHBuilderWithoutAngle4PixelTriplets_cfi import *
from CalibTracker.Configuration.SiStrip_FakeConditions_cff import *
from RecoHI.HiTracking.PixelLowPtTracksWithZPos_cff import *
import RecoHI.HiTracking.PixelLowPtTracksWithZPos_cfi
pixel3ProtoTracks = pixelLowPtTracksWithZPos.clone()
from RecoHI.HiTracking.PixelVertices_cfi import *
import RecoHI.HiTracking.PixelLowPtTracksWithZPos_cfi
pixel3PrimTracks = pixelLowPtTracksWithZPos.clone()
from RecoHI.HiTracking.PixelTrackSeeds_cfi import *
primSeeds = pixelTrackSeeds.clone()
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cff import *
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
ckfBaseTrajectoryFilterForMinBias = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
from RecoHI.HiTracking.TrajectoryCleanerBySharedSeeds_cfi import *
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
primTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
globalPrimTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
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

GlobalTrackingGeometryESProducer = cms.ESProducer("GlobalTrackingGeometryESProducer")

TransientTrackBuilderESProducer = cms.ESProducer("TransientTrackBuilderESProducer",
    ComponentName = cms.string('TransientTrackBuilder')
)

heavyIonTracking = cms.Sequence(pixel3ProtoTracks*pixelVertices*pixel3PrimTracks*primSeeds*primTrackCandidates*globalPrimTracks)
hiTrackingWithOfflineBeamSpot = cms.Sequence(offlineBeamSpot*trackerlocalreco*heavyIonTracking)
pixel3ProtoTracks.passLabel = 'Pixel triplet tracks for vertexing'
pixel3ProtoTracks.RegionFactoryPSet.RegionPSet.ptMin = 0.2
pixel3ProtoTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.1
pixelVertices.TrackCollection = 'pixel3ProtoTracks'
pixelVertices.PtMin = 0.5
pixel3PrimTracks.passLabel = 'Pixel triplet primary tracks with vertex constraint'
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.ptMin = 0.2
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.1
pixel3PrimTracks.RegionFactoryPSet.RegionPSet.originHalfLength = 0.1
primSeeds.tripletList = ['pixel3PrimTracks']
ckfBaseTrajectoryFilterForMinBias.ComponentName = 'ckfBaseTrajectoryFilterForMinBias'
ckfBaseTrajectoryFilterForMinBias.filterPset.minimumNumberOfHits = 6
ckfBaseTrajectoryFilterForMinBias.filterPset.minPt = 0.2
GroupedCkfTrajectoryBuilder.maxCand = 5
GroupedCkfTrajectoryBuilder.intermediateCleaning = False
GroupedCkfTrajectoryBuilder.alwaysUseInvalidHits = False
GroupedCkfTrajectoryBuilder.trajectoryFilterName = 'MinBiasCkfTrajectoryFilter'
MaterialPropagator.Mass = 0.139
OppositeMaterialPropagator.Mass = 0.139
primTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedSeeds'
#primTrackCandidates.SeedProducer = 'primSeeds'
primTrackCandidates.src = 'primSeeds' #change for 3_1_x
primTrackCandidates.RedundantSeedCleaner = 'none'
globalPrimTracks.src = 'primTrackCandidates'
globalPrimTracks.TrajectoryInEvent = True
# TrackAssociatorByHits.MinHitCut = 0.0


