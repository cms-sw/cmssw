import FWCore.ParameterSet.Config as cms

#################################
# Remaining clusters
thirdClusters = cms.EDProducer("TrackClusterRemover",
    oldClusterRemovalInfo = cms.InputTag("secondClusters"),
    trajectories = cms.InputTag("globalSecoTracks"),
    pixelClusters = cms.InputTag("secondClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(999999.0)
    ),
    stripClusters = cms.InputTag("secondClusters")
)

#################################
# Remaining pixel hits
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
thirdPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
thirdPixelRecHits.src = 'thirdClusters:'

#################################
# Remaining strip hits
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
thirdStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
thirdStripRecHits.ClusterProducer = 'thirdClusters'

#################################
# Tertiary pairs
ThirdLayerPairs = cms.EDProducer("SeedingLayersEDProducer",
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
        'BPix2+FPix2_neg'),
#       'FPix1_pos+FPix2_pos',
#       'FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('thirdPixelRecHits')
    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('thirdPixelRecHits')
    )
)

#################################
# Pixel-2 tertiary tracks
import RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi
pixelTertTracks = RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi.allPixelTracks.clone()
pixelTertTracks.passLabel = 'Pixel pair tracks with vertex constraint'
pixelTertTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.4 # 0.2
pixelTertTracks.RegionFactoryPSet.RegionPSet.useFoundVertices = True
pixelTertTracks.OrderedHitsFactoryPSet.ComponentName = 'StandardHitPairGenerator'
pixelTertTracks.OrderedHitsFactoryPSet.SeedingLayers = 'ThirdLayerPairs'
pixelTertTracks.OrderedHitsFactoryPSet.GeneratorPSet.ComponentName = 'StandardHitPairGenerator'
pixelTertTracks.FilterPSet = cms.PSet(
        ComponentName = cms.string('none')
    )

#################################
# Tertiary seeds
import RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi
tertSeeds = RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone()
tertSeeds.InputCollection = 'pixelTertTracks'

#################################
# Tertiary measurement tracker
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
thirdMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
thirdMeasurementTracker.ComponentName        = 'thirdMeasurementTracker'
thirdMeasurementTracker.pixelClusterProducer = 'thirdClusters'
thirdMeasurementTracker.stripClusterProducer = 'thirdClusters'

#################################
# Tertiary trajectory builder
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
thirdCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
thirdCkfTrajectoryBuilder.ComponentName          = 'thirdCkfTrajectoryBuilder'
thirdCkfTrajectoryBuilder.MeasurementTrackerName = 'thirdMeasurementTracker'
thirdCkfTrajectoryBuilder.trajectoryFilterName   = 'MinBiasCkfTrajectoryFilter'
thirdCkfTrajectoryBuilder.inOutTrajectoryFilterName   = 'MinBiasCkfTrajectoryFilter'

#################################
# Tertiary track candidates
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
tertTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
tertTrackCandidates.TrajectoryBuilder    = 'thirdCkfTrajectoryBuilder'
tertTrackCandidates.TrajectoryCleaner    = 'TrajectoryCleanerBySharedSeeds'
tertTrackCandidates.src                  = 'tertSeeds'
tertTrackCandidates.RedundantSeedCleaner = 'none'
tertTrackCandidates.useHitsSplitting          = cms.bool(False)
tertTrackCandidates.doSeedingRegionRebuilding = cms.bool(False)

#################################
# Global tertiary tracks
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
globalTertTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
globalTertTracks.clusterRemovalInfo = 'thirdClusters'
globalTertTracks.src                = 'tertTrackCandidates'
globalTertTracks.TrajectoryInEvent  = cms.bool(True)
