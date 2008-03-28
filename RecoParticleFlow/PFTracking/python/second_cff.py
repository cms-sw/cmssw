import FWCore.ParameterSet.Config as cms

import copy
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
#TRACKER HITS
secondPixelRecHits = copy.deepcopy(siPixelRecHits)
import copy
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
secondStripRecHits = copy.deepcopy(siStripMatchedRecHits)
import copy
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
#TRAJECTORY MEASUREMENT
secondMeasurementTracker = copy.deepcopy(MeasurementTracker)
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
#TRAJECTORY FILTER
secondCkfTrajectoryFilter = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi import *
#TRAJECTORY BUILDER
secondCkfTrajectoryBuilder = copy.deepcopy(GroupedCkfTrajectoryBuilder)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
#TRACK CANDIDATES
secondTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
#TRACKS
secondWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
#HIT REMOVAL
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
    layerList = cms.vstring('BPix1+BPix2+BPix3', 'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', 'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg'),
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
            useBending = cms.bool(True),
            useFixedPreFiltering = cms.bool(False),
            ComponentName = cms.string('PixelTripletHLTGenerator'),
            extraHitRPhitolerance = cms.double(0.06),
            useMultScattering = cms.bool(True),
            phiPreFiltering = cms.double(0.3),
            extraHitRZtolerance = cms.double(0.06)
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

