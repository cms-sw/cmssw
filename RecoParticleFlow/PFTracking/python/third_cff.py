import FWCore.ParameterSet.Config as cms

import copy
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
#TRACKER HITS
thirdPixelRecHits = copy.deepcopy(siPixelRecHits)
import copy
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
thirdStripRecHits = copy.deepcopy(siStripMatchedRecHits)
import copy
from RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cfi import *
thirdMixedSeeds = copy.deepcopy(globalMixedSeeds)
import copy
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
#TRAJECTORY MEASUREMENT
thirdMeasurementTracker = copy.deepcopy(MeasurementTracker)
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
#TRAJECTORY FILTER
thirdCkfTrajectoryFilter = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi import *
#TRAJECTORY BUILDER
thirdCkfTrajectoryBuilder = copy.deepcopy(GroupedCkfTrajectoryBuilder)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
#TRACK CANDIDATES
thirdTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
#TRACKS
thirdWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
#HIT REMOVAL
thirdClusters = cms.EDProducer("RemainingClusterProducer",
    matchedRecHits = cms.InputTag("secondStripRecHits","matchedRecHit"),
    recTracks = cms.InputTag("secondvtxFilt"),
    stereorecHits = cms.InputTag("secondStripRecHits","stereoRecHit"),
    rphirecHits = cms.InputTag("secondStripRecHits","rphiRecHit"),
    pixelHits = cms.InputTag("secondPixelRecHits")
)

#SEEDS
thirdlayerpairs = cms.ESProducer("MixedLayerPairsESProducer",
    ComponentName = cms.string('ThirdLayerPairs'),
    layerList = cms.vstring('BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3', 'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 'BPix1+FPix2_pos', 'BPix1+FPix2_neg', 'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 'BPix2+FPix2_pos', 'BPix2+FPix2_neg', 'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg', 'FPix2_pos+TEC1_pos', 'FPix2_pos+TEC2_pos', 'TEC2_pos+TEC3_pos', 'FPix2_neg+TEC1_neg', 'FPix2_neg+TEC2_neg', 'TEC2_neg+TEC3_neg'),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("thirdStripRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('thirdPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('thirdPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    )
)

thirdvtxFilt = cms.EDFilter("VertexFilter",
    recVertices = cms.InputTag("pixelVertices"),
    MinHits = cms.int32(5),
    DistRhoFromVertex = cms.double(0.1),
    DistZFromVertex = cms.double(0.1),
    recTracks = cms.InputTag("thirdWithMaterialTracks"),
    ChiCut = cms.double(250000.0),
    VertexCut = cms.bool(True)
)

third = cms.Sequence(thirdClusters*thirdPixelRecHits*thirdStripRecHits*thirdMixedSeeds*thirdTrackCandidates*thirdWithMaterialTracks*thirdvtxFilt)
thirdPixelRecHits.src = 'thirdClusters'
thirdStripRecHits.ClusterProducer = 'thirdClusters'
thirdMixedSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.5
thirdMixedSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'ThirdLayerPairs'
thirdMeasurementTracker.ComponentName = 'thirdMeasurementTracker'
thirdMeasurementTracker.pixelClusterProducer = 'thirdClusters'
thirdMeasurementTracker.stripClusterProducer = 'thirdClusters'
thirdCkfTrajectoryFilter.ComponentName = 'thirdCkfTrajectoryFilter'
thirdCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 3
thirdCkfTrajectoryFilter.filterPset.maxLostHits = 1
thirdCkfTrajectoryFilter.filterPset.minPt = 0.5
thirdCkfTrajectoryBuilder.ComponentName = 'thirdCkfTrajectoryBuilder'
thirdCkfTrajectoryBuilder.MeasurementTrackerName = 'thirdMeasurementTracker'
thirdCkfTrajectoryBuilder.trajectoryFilterName = 'thirdCkfTrajectoryFilter'
thirdTrackCandidates.SeedProducer = 'thirdMixedSeeds'
thirdTrackCandidates.TrajectoryBuilder = 'thirdCkfTrajectoryBuilder'
thirdWithMaterialTracks.src = 'thirdTrackCandidates'

