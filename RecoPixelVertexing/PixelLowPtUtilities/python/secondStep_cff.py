import FWCore.ParameterSet.Config as cms

import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
secondPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
secondStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
import RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi
pixelSecoTracks = RecoPixelVertexing.PixelLowPtUtilities.AllPixelTracks_cfi.allPixelTracks.clone()
import RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi
secoSeeds = RecoPixelVertexing.PixelLowPtUtilities.TrackSeeds_cfi.pixelTrackSeeds.clone()
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
secondMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
secondCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
secoTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
globalSecoTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
secondClusters = cms.EDFilter("TrackClusterRemover",
    trajectories = cms.InputTag("globalPrimTracks"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(999999.0)
    ),
    stripClusters = cms.InputTag("siStripClusters")
)

from RecoPixelVertexing.PixelLowPtUtilities.common_cff import BPixError
from RecoPixelVertexing.PixelLowPtUtilities.common_cff import FPixError

secondLayerTriplets = cms.ESProducer("PixelLayerTripletsESProducer",
    ComponentName = cms.string('SecondLayerTriplets'),
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        BPixError,
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('secondPixelRecHits')
    ),
    FPix = cms.PSet(
        FPixError,
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('secondPixelRecHits')
    )
)

secondPixelRecHits.src = 'secondClusters:'
secondStripRecHits.ClusterProducer = 'secondClusters'
pixelSecoTracks.passLabel = 'Pixel triplet tracks without vertex constraint'
pixelSecoTracks.RegionFactoryPSet.RegionPSet.originRadius = 3.5
pixelSecoTracks.OrderedHitsFactoryPSet.SeedingLayers = 'SecondLayerTriplets'
secoSeeds.tripletList = ['pixelSecoTracks']
secondMeasurementTracker.ComponentName = 'secondMeasurementTracker'
secondMeasurementTracker.pixelClusterProducer = 'secondClusters'
secondMeasurementTracker.stripClusterProducer = 'secondClusters'
secondCkfTrajectoryBuilder.ComponentName = 'secondCkfTrajectoryBuilder'
secondCkfTrajectoryBuilder.MeasurementTrackerName = 'secondMeasurementTracker'
secondCkfTrajectoryBuilder.trajectoryFilterName = 'MinBiasCkfTrajectoryFilter'

secoTrackCandidates.TrajectoryCleaner = 'TrajectoryCleanerBySharedSeeds'
secoTrackCandidates.src = 'secoSeeds'
secoTrackCandidates.TrajectoryBuilder = 'secondCkfTrajectoryBuilder'
secoTrackCandidates.RedundantSeedCleaner = 'none'
secoTrackCandidates.doSeedingRegionRebuilding = False

globalSecoTracks.src = 'secoTrackCandidates'
globalSecoTracks.clusterRemovalInfo = 'secondClusters'
globalSecoTracks.TrajectoryInEvent = True

