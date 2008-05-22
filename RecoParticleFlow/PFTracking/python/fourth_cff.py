import FWCore.ParameterSet.Config as cms

from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
#TRACKER HITS
fourthPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
fourthStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cfi
fourthMixedSeeds = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cfi.globalMixedSeeds.clone()
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
#TRAJECTORY MEASUREMENT
fourthMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
#TRAJECTORY FILTER
fourthCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
#TRAJECTORY BUILDER
fourthCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
#TRACK CANDIDATES
fourthTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
#TRACKS
fourthWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
#HIT REMOVAL
fourthClusters = cms.EDProducer("RemainingClusterProducer",
    matchedRecHits = cms.InputTag("thirdStripRecHits","matchedRecHit"),
    recTracks = cms.InputTag("thirdvtxFilt"),
    stereorecHits = cms.InputTag("thirdStripRecHits","stereoRecHit"),
    rphirecHits = cms.InputTag("thirdStripRecHits","rphiRecHit"),
    pixelHits = cms.InputTag("thirdPixelRecHits")
)

#SEEDS
fourthlayerpairs = cms.ESProducer("MixedLayerPairsESProducer",
    ComponentName = cms.string('FourthLayerPairs'),
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
        'FPix1_neg+FPix2_neg', 
        'FPix2_pos+TEC1_pos', 
        'FPix2_pos+TEC2_pos', 
        'TEC2_pos+TEC3_pos', 
        'FPix2_neg+TEC1_neg', 
        'FPix2_neg+TEC2_neg', 
        'TEC2_neg+TEC3_neg'),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("fourthStripRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('fourthPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('fourthPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    )
)

fourthvtxFilt = cms.EDFilter("VertexFilter",
    recVertices = cms.InputTag("pixelVertices"),
    MinHits = cms.int32(3),
    DistRhoFromVertex = cms.double(0.1),
    DistZFromVertex = cms.double(0.1),
    recTracks = cms.InputTag("fourthWithMaterialTracks"),
    ChiCut = cms.double(250000.0),
    VertexCut = cms.bool(True)
)

fourth = cms.Sequence(fourthClusters*fourthPixelRecHits*fourthStripRecHits*fourthMixedSeeds*fourthTrackCandidates*fourthWithMaterialTracks*fourthvtxFilt)
fourthPixelRecHits.src = cms.InputTag("fourthClusters")
fourthStripRecHits.ClusterProducer = 'fourthClusters'
fourthMixedSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'FourthLayerPairs'
fourthMixedSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.3
fourthMeasurementTracker.ComponentName = 'fourthMeasurementTracker'
fourthMeasurementTracker.pixelClusterProducer = 'fourthClusters'
fourthMeasurementTracker.stripClusterProducer = 'fourthClusters'
fourthCkfTrajectoryFilter.ComponentName = 'fourthCkfTrajectoryFilter'
fourthCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 3
fourthCkfTrajectoryFilter.filterPset.maxLostHits = 1
fourthCkfTrajectoryFilter.filterPset.minPt = 0.3
fourthCkfTrajectoryBuilder.ComponentName = 'fourthCkfTrajectoryBuilder'
fourthCkfTrajectoryBuilder.MeasurementTrackerName = 'fourthMeasurementTracker'
fourthCkfTrajectoryBuilder.trajectoryFilterName = 'fourthCkfTrajectoryFilter'
fourthTrackCandidates.SeedProducer = 'fourthMixedSeeds'
fourthTrackCandidates.TrajectoryBuilder = 'fourthCkfTrajectoryBuilder'
fourthWithMaterialTracks.src = 'fourthTrackCandidates'

