import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.PixelLessStep_cff import *

# SEEDING LAYERS
pixelLessStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring(
    #TIB
    'TIB1+TIB2+MTIB3',
    #TIB+TID
    'TIB1+TIB2+MTID1_pos','TIB1+TIB2+MTID1_neg',
    #TID
    'TID1_pos+TID2_pos+MTID3_pos','TID1_neg+TID2_neg+MTID3_neg',#ring 3 (mono)
    'TID1_pos+TID2_pos+TID3_pos','TID1_neg+TID2_neg+TID3_neg',#ring 1-2 (matched)
    #TID+TEC RING 1-3
    'TID2_pos+TID3_pos+TEC1_pos','TID2_neg+TID3_neg+TEC1_neg',
    'TID3_pos+TEC1_pos+TEC2_pos','TID3_neg+TEC1_neg+TEC2_neg',
    #TEC RING 1-3
    'TEC1_pos+TEC2_pos+TEC3_pos','TEC1_neg+TEC2_neg+TEC3_neg',
    'TEC2_pos+TEC3_pos+TEC4_pos','TEC2_neg+TEC3_neg+TEC4_neg',
    'TEC3_pos+TEC4_pos+TEC5_pos','TEC3_neg+TEC4_neg+TEC5_neg',
    'TEC2_pos+TEC3_pos+TEC5_pos','TEC2_neg+TEC3_neg+TEC5_neg'
    ),
    TIB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'),
         matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
         skipClusters   = cms.InputTag('pixelLessStepClusters')
    ),
    MTIB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'),
         skipClusters   = cms.InputTag('pixelLessStepClusters'),
         rphiRecHits    = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        skipClusters = cms.InputTag('pixelLessStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    MTID = cms.PSet(
        rphiRecHits    = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        skipClusters = cms.InputTag('pixelLessStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(3),
        maxRing = cms.int32(3)
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        skipClusters = cms.InputTag('pixelLessStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    )
)
# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
pixelLessStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
pixelLessStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'pixelLessStepSeedLayers'
pixelLessStepSeeds.OrderedHitsFactoryPSet.ComponentName = 'StandardMultiHitGenerator'
pixelLessStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(
    useFixedPreFiltering = cms.bool(False),
    maxElement = cms.uint32(100000),
    ComponentName = cms.string('MultiHitGeneratorFromChi2'),
    extraHitRPhitolerance = cms.double(0.2),
    phiPreFiltering = cms.double(0.3),
    extraHitRZtolerance = cms.double(0.25),
    fnSigmaRZ = cms.double(2.0),
    chi2VsPtCut = cms.bool(True),
    pt_interv = cms.vdouble(0.7,1.0,2.0,5.0),
    chi2_cuts = cms.vdouble(),
    maxChi2 = cms.double(4.0),
    extraPhiKDBox = cms.double(0.0),
    SimpleMagneticField = cms.string(''),
#    SimpleMagneticField = cms.string('ParabolicMf'),
    refitHits = cms.bool(True),
    ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
    debug = cms.bool(False),
    detIdsToDebug = cms.vint32(0,0,0)
)
pixelLessStepSeeds.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.4
pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 15.0
pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 2.5
#pixelLessStepSeeds.SeedCreatorPSet.OriginTransverseErrorMultiplier = 1.0 #2.0 this is not used according to kevin
pixelLessStepSeeds.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(False),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
)

PixelLessStep = cms.Sequence(
    pixelLessStepClusters*
    pixelLessStepSeedLayers*
    pixelLessStepSeeds*
    pixelLessStepTrackCandidates*
    pixelLessStepTracks*
    pixelLessStepSelector
)
