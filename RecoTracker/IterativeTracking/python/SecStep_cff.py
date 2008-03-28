import FWCore.ParameterSet.Config as cms

import copy
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
#TRACKER HITS
secPixelRecHits = copy.deepcopy(siPixelRecHits)
import copy
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
secStripRecHits = copy.deepcopy(siStripMatchedRecHits)
import copy
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTripletsWithVertices_cfi import *
#SEEDS
#TRIPLETS
secTriplets = copy.deepcopy(globalSeedsFromTripletsWithVertices)
import copy
from RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi import *
#TRAJECTORY MEASUREMENT
secMeasurementTracker = copy.deepcopy(MeasurementTracker)
import copy
from TrackingTools.TrajectoryFiltering.TrajectoryFilterESProducer_cfi import *
#TRAJECTORY FILTER
secCkfTrajectoryFilter = copy.deepcopy(trajectoryFilterESProducer)
import copy
from RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilderESProducer_cfi import *
#TRAJECTORY BUILDER
secCkfTrajectoryBuilder = copy.deepcopy(GroupedCkfTrajectoryBuilder)
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
#TRACK CANDIDATES
secTrackCandidates = copy.deepcopy(ckfTrackCandidates)
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
#TRACKS
secWithMaterialTracks = copy.deepcopy(ctfWithMaterialTracks)
#HIT REMOVAL
#module secClusters = RemainingClusterProducer{
#    InputTag matchedRecHits    = siStripMatchedRecHits:matchedRecHit
#    InputTag rphirecHits       = siStripMatchedRecHits:rphiRecHit
#    InputTag stereorecHits     = siStripMatchedRecHits:stereoRecHit
#    InputTag pixelHits         = siPixelRecHits:
#    InputTag recTracks         = firstfilter:
#}
secClusters = cms.EDFilter("TrackClusterRemover",
    trajectories = cms.InputTag("firstfilter"),
    pixelClusters = cms.InputTag("siPixelClusters"),
    Common = cms.PSet(
        maxChi2 = cms.double(30.0)
    ),
    stripClusters = cms.InputTag("siStripClusters")
)

#SEEDING LAYERS
#PAIRS
seclayerPairs = cms.ESProducer("MixedLayerPairsESProducer",
    TIB3 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        rphiRecHits = cms.InputTag("secStripRecHits","rphiRecHit")
    ),
    TIB2 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        matchedRecHits = cms.InputTag("secStripRecHits","matchedRecHit"),
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        rphiRecHits = cms.InputTag("secStripRecHits","rphiRecHit")
    ),
    TIB1 = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        matchedRecHits = cms.InputTag("secStripRecHits","matchedRecHit"),
        useSimpleRphiHitsCleaner = cms.untracked.bool(False),
        rphiRecHits = cms.InputTag("secStripRecHits","rphiRecHit")
    ),
    ComponentName = cms.string('SecLayerPairs'),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("secStripRecHits","matchedRecHit"),
        useRingSlector = cms.untracked.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(1)
    ),
    layerList = cms.vstring('BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3', 'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 'BPix1+FPix2_pos', 'BPix1+FPix2_neg', 'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 'BPix2+FPix2_pos', 'BPix2+FPix2_neg', 'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg', 'FPix2_pos+TEC1_pos', 'FPix2_pos+TEC2_pos', 'TEC1_pos+TEC2_pos', 'TEC2_pos+TEC3_pos', 'FPix2_neg+TEC1_neg', 'FPix2_neg+TEC2_neg', 'TEC1_neg+TEC2_neg', 'TEC2_neg+TEC3_neg', 'TIB1+TIB2+TIB3'),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('secPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    ),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4MixedPairs'),
        HitProducer = cms.string('secPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    )
)

#TRIPLETS
seclayertriplets = cms.ESProducer("PixelLayerTripletsESProducer",
    ComponentName = cms.string('SecLayerTriplets'),
    layerList = cms.vstring('BPix1+BPix2+BPix3', 'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', 'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0027),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('secPixelRecHits'),
        hitErrorRZ = cms.double(0.006)
    ),
    FPix = cms.PSet(
        useErrorsFromParam = cms.untracked.bool(True),
        hitErrorRPhi = cms.double(0.0051),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
        HitProducer = cms.string('secPixelRecHits'),
        hitErrorRZ = cms.double(0.0036)
    )
)

secStep = cms.EDFilter("VertexFilter",
    TrackAlgorithm = cms.string('iter2'),
    recVertices = cms.InputTag("pixelVertices"),
    MinHits = cms.int32(3),
    DistRhoFromVertex = cms.double(0.1),
    DistZFromVertex = cms.double(0.4),
    recTracks = cms.InputTag("secWithMaterialTracks"),
    UseQuality = cms.bool(True),
    ChiCut = cms.double(130.0),
    TrackQuality = cms.string('highPurity'),
    VertexCut = cms.bool(True)
)

secondStep = cms.Sequence(secClusters*secPixelRecHits*secStripRecHits*secTriplets*secTrackCandidates*secWithMaterialTracks*secStep)
secPixelRecHits.src = cms.InputTag("secClusters")
secStripRecHits.ClusterProducer = 'secClusters'
secTriplets.RegionFactoryPSet.RegionPSet.originHalfLength = 22.7
secTriplets.OrderedHitsFactoryPSet.SeedingLayers = 'SecLayerTriplets'
secTriplets.RegionFactoryPSet.RegionPSet.ptMin = 0.3
secMeasurementTracker.ComponentName = 'secMeasurementTracker'
secMeasurementTracker.pixelClusterProducer = 'secClusters'
secMeasurementTracker.stripClusterProducer = 'secClusters'
secCkfTrajectoryFilter.ComponentName = 'secCkfTrajectoryFilter'
secCkfTrajectoryFilter.filterPset.maxLostHits = 1
secCkfTrajectoryFilter.filterPset.minimumNumberOfHits = 3
secCkfTrajectoryFilter.filterPset.minPt = 0.3
secCkfTrajectoryBuilder.ComponentName = 'secCkfTrajectoryBuilder'
secCkfTrajectoryBuilder.MeasurementTrackerName = 'secMeasurementTracker'
secCkfTrajectoryBuilder.trajectoryFilterName = 'secCkfTrajectoryFilter'
secTrackCandidates.SeedProducer = 'secTriplets'
secTrackCandidates.TrajectoryBuilder = 'secCkfTrajectoryBuilder'
secWithMaterialTracks.src = 'secTrackCandidates'
secWithMaterialTracks.clusterRemovalInfo = 'secClusters'

