import FWCore.ParameterSet.Config as cms

import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
thirdPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
thirdStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
import RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi
pixel2PrimTracks = RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi.allPixelTracks.clone()
import RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi
tertSeeds = RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone()
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
thirdMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
thirdCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
tertTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
globalTertTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
thirdClusters = cms.EDFilter("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("secondClusters"),
    trajectories = cms.InputTag("globalSecoTracks"),
    pixelClusters = cms.InputTag("secondClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(999999.0)
    ),
    stripClusters = cms.InputTag("secondClusters")
)

from RecoPixelVertexing.PixelLowPtUtilities.common_cff import BPixError
from RecoPixelVertexing.PixelLowPtUtilities.common_cff import FPixError

thirdLayerPairs = cms.ESProducer("PixelLayerPairsESProducer",
    ComponentName = cms.string('thirdLayerPairs'),
    layerList = cms.vstring('BPix1+BPix2', 
        'BPix1+BPix3', 
        'BPix2+BPix3', 
        'BPix1+FPix1_pos', 
        'BPix1+FPix1_neg', 
        'BPix1+FPix2_pos', 
        'BPix1+FPix2_neg', 
        'BPix2+FPix1_pos', 
        'BPix2+FPix1_neg', 
        'BPix2+FPix2_pos', 
        'BPix2+FPix2_neg', 
        'FPix1_pos+FPix2_pos', 
        'FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        BPixError,
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('thirdPixelRecHits')
    ),
    FPix = cms.PSet(
        FPixError,
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('thirdPixelRecHits')
    )
)

thirdPixelRecHits.src = 'thirdClusters:'
thirdStripRecHits.ClusterProducer = 'thirdClusters'
pixel2PrimTracks.passLabel = 'Pixel pair tracks with vertex constraint'
pixel2PrimTracks.RegionFactoryPSet.RegionPSet.useFoundVertices = True
#pixel2PrimTracks.RegionFactoryPSet.RegionPSet.fixedError = 0.2
pixel2PrimTracks.OrderedHitsFactoryPSet.ComponentName = 'StandardHitPairGenerator'
pixel2PrimTracks.OrderedHitsFactoryPSet.SeedingLayers = 'thirdLayerPairs'
pixel2PrimTracks.OrderedHitsFactoryPSet.GeneratorPSet.ComponentName = 'StandardHitPairGenerator'
tertSeeds.tripletList = ['pixel2PrimTracks']
thirdMeasurementTracker.ComponentName = 'thirdMeasurementTracker'
thirdMeasurementTracker.pixelClusterProducer = 'thirdClusters'
thirdMeasurementTracker.stripClusterProducer = 'thirdClusters'
thirdCkfTrajectoryBuilder.ComponentName = 'thirdCkfTrajectoryBuilder'
thirdCkfTrajectoryBuilder.MeasurementTrackerName = 'thirdMeasurementTracker'
thirdCkfTrajectoryBuilder.trajectoryFilterName = 'MinBiasCkfTrajectoryFilter'

tertTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedSeeds'
tertTrackCandidates.src = 'tertSeeds'
tertTrackCandidates.TrajectoryBuilder = 'thirdCkfTrajectoryBuilder'
tertTrackCandidates.RedundantSeedCleaner = 'none'
tertTrackCandidates.doSeedingRegionRebuilding = False

globalTertTracks.src = 'tertTrackCandidates'
globalTertTracks.clusterRemovalInfo = 'thirdClusters'
globalTertTracks.TrajectoryInEvent = True

