import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.PostLS1_PixelLessStep_cff import *

# SEEDING LAYERS BARREL
pixelLessStepSeedLayersA = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring(
    #TIB
    'TIB1+TIB2+MTIB3',
    #TIB+TID
    'TIB1+TIB2+MTID1_pos','TIB1+TIB2+MTID1_neg'
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
    MTID = cms.PSet(
        rphiRecHits    = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        skipClusters = cms.InputTag('pixelLessStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(3),
        maxRing = cms.int32(3)
    )
)
# SEEDS BARREL
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
pixelLessStepSeedsA = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
pixelLessStepSeedsA.OrderedHitsFactoryPSet.SeedingLayers = 'pixelLessStepSeedLayersA'
pixelLessStepSeedsA.OrderedHitsFactoryPSet.ComponentName = 'StandardMultiHitGenerator'
pixelLessStepSeedsA.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(
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
    maxChi2 = cms.double(4.0),#3.0 in v3
    refitHits = cms.bool(True),
    extraPhiKDBox = cms.double(0.),
    SimpleMagneticField = cms.string(''),
#    SimpleMagneticField = cms.string('ParabolicMf'),
    ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
    debug = cms.bool(False),
    detIdsToDebug = cms.vint32(0,0,0)
)
pixelLessStepSeedsA.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
pixelLessStepSeedsA.RegionFactoryPSet.RegionPSet.ptMin = 0.55
pixelLessStepSeedsA.RegionFactoryPSet.RegionPSet.originHalfLength = 12.0
pixelLessStepSeedsA.RegionFactoryPSet.RegionPSet.originRadius = 1.
pixelLessStepSeedsA.SeedCreatorPSet.OriginTransverseErrorMultiplier = 1.0
pixelLessStepSeedsA.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(False),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
)
 
# SEEDING LAYERS ENDCAP
pixelLessStepSeedLayersB = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring(
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
# SEEDS ENDCAP
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
pixelLessStepSeedsB = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
pixelLessStepSeedsB.OrderedHitsFactoryPSet.SeedingLayers = 'pixelLessStepSeedLayersB'
pixelLessStepSeedsB.OrderedHitsFactoryPSet.ComponentName = 'StandardMultiHitGenerator'
pixelLessStepSeedsB.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(
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
    maxChi2 = cms.double(6.0),#5.0 in v3 #4.0 in v2
    extraPhiKDBox = cms.double(0.),
    SimpleMagneticField = cms.string(''),
#    SimpleMagneticField = cms.string('ParabolicMf'),
    refitHits = cms.bool(True),
    ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
    debug = cms.bool(False),
    detIdsToDebug = cms.vint32(0,0,0)
)
pixelLessStepSeedsB.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
pixelLessStepSeedsB.RegionFactoryPSet.RegionPSet.ptMin = 0.4 #0.5 in v2
pixelLessStepSeedsB.RegionFactoryPSet.RegionPSet.originHalfLength = 15.0
pixelLessStepSeedsB.RegionFactoryPSet.RegionPSet.originRadius = 1.5
pixelLessStepSeedsB.SeedCreatorPSet.OriginTransverseErrorMultiplier = 1.0
pixelLessStepSeedsB.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(False),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
)

PixelLessStep = cms.Sequence(pixelLessStepClusters*
                             pixelLessStepSeedLayersA*
                             pixelLessStepSeedsA*
                             pixelLessStepSeedLayersB*
                             pixelLessStepSeedsB*
                             pixelLessStepSeeds*
                             pixelLessStepTrackCandidates*
                             pixelLessStepTracks*
                             pixelLessStepSelector)
