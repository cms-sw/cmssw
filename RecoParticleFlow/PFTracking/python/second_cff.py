import FWCore.ParameterSet.Config as cms

#HIT REMOVAL
from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
import RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi
#TRACKER HITS
secondPixelRecHits = RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi.siPixelRecHits.clone()
import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi
secondStripRecHits = RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone()
import RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi
#TRAJECTORY MEASUREMENT
secondMeasurementTracker = RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi.MeasurementTracker.clone()
import TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi
#TRAJECTORY FILTER
secondCkfTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi.trajectoryFilterESProducer.clone()
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi
#TRAJECTORY BUILDER
secondCkfTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi.GroupedCkfTrajectoryBuilder.clone()
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
#TRACK CANDIDATES
secondTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone()
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
#TRACKS
secondWithMaterialTracks = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
secondClusters = cms.EDProducer("RemainingClusterProducer",
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    recTracks = cms.InputTag("firstvtxFilt"),
    stereorecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
    rphirecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
    pixelHits = cms.InputTag("siPixelRecHits")
)

#SEEDS
secondlayertriplets = cms.ESProducer("PixelLayerTripletsESProducer",
    ComponentName = cms.string('SecondLayerTriplets'),
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('secondPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('secondPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    )
)

secondTriplets = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.string('SecondLayerTriplets'),
        GeneratorPSet = cms.PSet(
            PixelTripletHLTGenerator
        )
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('GlobalRegionProducer'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            originHalfLength = cms.double(15.9),
            originRadius = cms.double(0.2),
            originYPos = cms.double(0.0),
            ptMin = cms.double(0.5),
            originXPos = cms.double(0.0),
            originZPos = cms.double(0.0)
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)

secondvtxFilt = cms.EDFilter("VertexFilter",
    recVertices = cms.InputTag("pixelVertices"),
    MinHits = cms.int32(4),
    DistRhoFromVertex = cms.double(0.1),
    DistZFromVertex = cms.double(0.4),
    recTracks = cms.InputTag("secondWithMaterialTracks"),
    ChiCut = cms.double(130.0),
    VertexCut = cms.bool(True)
)

second = cms.Sequence(secondClusters*secondPixelRecHits*secondStripRecHits*secondTriplets*secondTrackCandidates*secondWithMaterialTracks*secondvtxFilt)
secondPixelRecHits.src = cms.InputTag("secondClusters")
secondStripRecHits.ClusterProducer = 'secondClusters'
secondMeasurementTracker.ComponentName = 'secondMeasurementTracker'
secondMeasurementTracker.pixelClusterProducer = 'secondClusters'
secondMeasurementTracker.stripClusterProducer = 'secondClusters'
secondCkfTrajectoryFilter.ComponentName = 'secondCkfTrajectoryFilter'
secondCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 4
secondCkfTrajectoryFilter.filterPset.maxLostHits = 1
secondCkfTrajectoryFilter.filterPset.minPt = 0.5
secondCkfTrajectoryBuilder.ComponentName = 'secondCkfTrajectoryBuilder'
secondCkfTrajectoryBuilder.MeasurementTrackerName = 'secondMeasurementTracker'
secondCkfTrajectoryBuilder.trajectoryFilterName = 'secondCkfTrajectoryFilter'
secondTrackCandidates.SeedProducer = 'secondTriplets'
secondTrackCandidates.TrajectoryBuilder = 'secondCkfTrajectoryBuilder'
secondWithMaterialTracks.src = 'secondTrackCandidates'

