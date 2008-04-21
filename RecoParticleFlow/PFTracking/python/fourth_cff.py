import FWCore.ParameterSet.Config as cms

from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilderWithoutRefit_cfi import *
import copy
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
#TRACKER HITS
fourthPixelRecHits = copy.deepcopy(siPixelRecHits)
import copy
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
fourthStripRecHits = copy.deepcopy(siStripMatchedRecHits)
import copy
from RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cfi import *
fourthMixedSeeds = copy.deepcopy(globalMixedSeeds)
import copy
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
#TRAJECTORY MEASUREMENT
fourthMeasurementTracker = copy.deepcopy(MeasurementTracker)
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
#TRAJECTORY FILTER
fourthCkfTrajectoryFilter = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi import *
#TRAJECTORY BUILDER
fourthCkfTrajectoryBuilder = copy.deepcopy(GroupedCkfTrajectoryBuilder)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
#TRACK CANDIDATES
fourthTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
#TRACKS
fourthWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
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

