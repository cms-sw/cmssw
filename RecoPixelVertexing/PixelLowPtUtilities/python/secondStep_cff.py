import FWCore.ParameterSet.Config as cms

#################################
# Remaining clusters
secondClusters = cms.EDProducer("TrackClusterRemover",
    trajectories = cms.InputTag("globalPrimTracks"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(999999.0)
    ),
    stripClusters = cms.InputTag("siStripClusters")
)

#################################
# Remaining pixel hits
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
secondPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
secondPixelRecHits.src = 'secondClusters:'

#################################
# Remaining strip hits
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
secondStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
secondStripRecHits.ClusterProducer = 'secondClusters'

#################################
# Secondary triplets
SecondLayerTriplets = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2+BPix3',
        'BPix1+BPix2+FPix1_pos',
        'BPix1+BPix2+FPix1_neg',
        'BPix1+FPix1_pos+FPix2_pos',
        'BPix1+FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('secondPixelRecHits')
    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('secondPixelRecHits')
    )
)

#################################
# Pixel-3 secondary tracks
import RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi
pixelSecoTracks = RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi.allPixelTracks.clone()
pixelSecoTracks.passLabel = 'Pixel triplet tracks without vertex constraint'
pixelSecoTracks.RegionFactoryPSet.RegionPSet.originRadius = 3.5
pixelSecoTracks.OrderedHitsFactoryPSet.SeedingLayers = 'SecondLayerTriplets'

#################################
# Secondary seeds
import RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi
secoSeeds = RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone()
secoSeeds.InputCollection = 'pixelSecoTracks'

#################################
# Secondary measurement tracker
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
secondMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
secondMeasurementTracker.ComponentName        = 'secondMeasurementTracker'
secondMeasurementTracker.pixelClusterProducer = 'secondClusters'
secondMeasurementTracker.stripClusterProducer = 'secondClusters'

#################################
# Secondary trajectory builder
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
secondCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
secondCkfTrajectoryBuilder.ComponentName          = 'secondCkfTrajectoryBuilder'
secondCkfTrajectoryBuilder.MeasurementTrackerName = 'secondMeasurementTracker'
secondCkfTrajectoryBuilder.trajectoryFilterName   = 'MinBiasCkfTrajectoryFilter'
secondCkfTrajectoryBuilder.inOutTrajectoryFilterName   = 'MinBiasCkfTrajectoryFilter'

#################################
# Secondary track candidates
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
secoTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
secoTrackCandidates.TrajectoryBuilder    = 'secondCkfTrajectoryBuilder'
secoTrackCandidates.TrajectoryCleaner    = 'TrajectoryCleanerBySharedSeeds'
secoTrackCandidates.src                  = 'secoSeeds'
secoTrackCandidates.RedundantSeedCleaner = 'none'
secoTrackCandidates.useHitsSplitting          = cms.bool(False)
secoTrackCandidates.doSeedingRegionRebuilding = cms.bool(False)

#################################
# Global secondary tracks
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
globalSecoTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
globalSecoTracks.clusterRemovalInfo = 'secondClusters'
globalSecoTracks.src                = 'secoTrackCandidates'
globalSecoTracks.TrajectoryInEvent  = cms.bool(True)
