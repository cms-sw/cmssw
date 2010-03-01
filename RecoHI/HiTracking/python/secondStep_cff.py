import FWCore.ParameterSet.Config as cms

#################################
# Remaining clusters
hiNewClusters = cms.EDProducer("TrackClusterRemover",
    trajectories = cms.InputTag("hiSelectedTracks"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    stripClusters = cms.InputTag("siStripClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.) # remove none=0, remove all=9999
    )
)

# Remake corresponding rechits
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
hiNewPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone(
    src = cms.InputTag('hiNewClusters')
    )
hiNewStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
    ClusterProducer = 'hiNewClusters'
    )

#################################
# Pixel Pair Layers
import RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi
hiNewPixelLayerPairs = RecoTracker.TkSeedingLayers.PixelLayerPairs_cfi.pixellayerpairs.clone(
    ComponentName = 'hiNewPixelLayerPairs',
    )
hiNewPixelLayerPairs.BPix.HitProducer = 'hiNewPixelRecHits'
hiNewPixelLayerPairs.FPix.HitProducer = 'hiNewPixelRecHits'

# Pixel Pair Seeding
from RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff import *
hiNewSeedFromPairs = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff.globalSeedsFromPairsWithVertices.clone()
hiNewSeedFromPairs.RegionFactoryPSet.RegionPSet.ptMin = 10.0
hiNewSeedFromPairs.RegionFactoryPSet.RegionPSet.originRadius = 0.005
hiNewSeedFromPairs.RegionFactoryPSet.RegionPSet.fixedError=0.005
hiNewSeedFromPairs.RegionFactoryPSet.RegionPSet.VertexCollection=cms.InputTag("hiSelectedVertex")
hiNewSeedFromPairs.OrderedHitsFactoryPSet.SeedingLayers = cms.string('hiNewPixelLayerPairs')
hiNewSeedFromPairs.ClusterCheckPSet.doClusterCheck=False

#################################
# building 
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
hiNewMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone(
    ComponentName = 'hiNewMeasurementTracker',
    pixelClusterProducer = 'hiNewClusters',
    stripClusterProducer = 'hiNewClusters'
    )

import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
hiNewTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone(
    ComponentName = 'hiNewTrajectoryFilter',
    filterPset = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.filterPset.clone(
    minimumNumberOfHits = 6,
    minPt = 10.0
    )
    )

import RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi
hiNewCkfTrajectoryBuilder = RecoTracker.CkfPattern.CkfTrajectoryBuilderESProducer_cfi.CkfTrajectoryBuilder.clone(
    ComponentName = 'hiNewCkfTrajectoryBuilder',
    MeasurementTrackerName = 'hiNewMeasurementTracker',
    trajectoryFilterName = 'hiNewTrajectoryFilter',
    intermediateCleaning = False,
    alwaysUseInvalidHits = False
    )

import RecoHI.HiTracking.HICkfTrackCandidates_cff
hiNewTrackCandidates = RecoHI.HiTracking.HICkfTrackCandidates_cff.hiPrimTrackCandidates.clone(
    src = cms.InputTag('hiNewSeedFromPairs'),
    TrajectoryBuilder = 'hiNewCkfTrajectoryBuilder'
    )

#fitting
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
hiNewGlobalTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
hiNewGlobalTracks.clusterRemovalInfo = 'hiNewClusters'
hiNewGlobalTracks.src                = 'hiNewTrackCandidates'
hiNewGlobalTracks.TrajectoryInEvent  = cms.bool(True)

#selection
import RecoHI.HiTracking.HISelectedTracks_cfi
hiNewSelectedTracks = RecoHI.HiTracking.HISelectedTracks_cfi.hiSelectedTracks.clone(
    src = cms.InputTag("hiNewGlobalTracks"),
    max_z0 = 1.0
    )



#################################

secondStep = cms.Sequence(hiNewClusters * hiNewPixelRecHits * hiNewStripRecHits * hiNewSeedFromPairs *
                          hiNewTrackCandidates * hiNewGlobalTracks * hiNewSelectedTracks)
