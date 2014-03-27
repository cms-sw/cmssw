import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.TobTecStep_cff import *

# TRIPLET SEEDING LAYERS
tobTecStepSeedLayersTripl = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring(
    #TOB
    'TOB1+TOB2+MTOB3',
    #TOB+TEC
    'TOB1+TOB2+MTEC1_pos','TOB1+TOB2+MTEC1_neg',
    #'ITOB1+TEC1_pos+MTEC2_pos','ITOB1+TEC1_neg+MTEC2_neg',
    #TEC
    #'TEC1_pos+TEC2_pos+MTEC3_pos','TEC1_neg+TEC2_neg+MTEC3_neg',
    #'TEC2_pos+TEC3_pos+MTEC4_pos','TEC2_neg+TEC3_neg+MTEC4_neg',
    #'TEC3_pos+TEC4_pos+MTEC5_pos','TEC3_neg+TEC4_neg+MTEC5_neg',
    #'TEC4_pos+TEC5_pos+MTEC6_pos','TEC4_neg+TEC5_neg+MTEC6_neg',
    #'TEC5_pos+TEC6_pos+MTEC7_pos','TEC5_neg+TEC6_neg+MTEC7_neg',
    #'TEC6_pos+TEC7_pos+MTEC8_pos','TEC6_neg+TEC7_neg+MTEC8_neg',
    #'TEC7_pos+TEC8_pos+MTEC9_pos','TEC7_neg+TEC8_neg+MTEC9_neg' 
    ),
    TOB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'),
         matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
         skipClusters   = cms.InputTag('tobTecStepClusters')
    ),
    #ITOB = cms.PSet(
    #     TTRHBuilder    = cms.string('WithTrackAngle'),
    #     matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    #     skipClusters   = cms.InputTag('myTobTecHybridStepClusters'),
    #     MinAbsZ = cms.double(85.0)
    #),
    MTOB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'),
         skipClusters   = cms.InputTag('tobTecStepClusters'),
         rphiRecHits    = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    #TEC = cms.PSet(
    #    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    #    skipClusters = cms.InputTag('tobTecStepClusters'),
    #    useRingSlector = cms.bool(True),
    #    TTRHBuilder = cms.string('WithTrackAngle'),
    #    minRing = cms.int32(5),
    #    maxRing = cms.int32(5)
    #),
    MTEC = cms.PSet(
        rphiRecHits    = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        skipClusters = cms.InputTag('tobTecStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(6),
        maxRing = cms.int32(6)
    )
)
# TRIPLET SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
tobTecStepSeedsTripl = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
tobTecStepSeedsTripl.OrderedHitsFactoryPSet.SeedingLayers = 'tobTecStepSeedLayersTripl'
tobTecStepSeedsTripl.OrderedHitsFactoryPSet.ComponentName = 'StandardMultiHitGenerator'
tobTecStepSeedsTripl.OrderedHitsFactoryPSet.GeneratorPSet = cms.PSet(
    useFixedPreFiltering = cms.bool(False),
    maxElement = cms.uint32(100000),
    ComponentName = cms.string('MultiHitGeneratorFromChi2'),
    extraHitRPhitolerance = cms.double(0.2),
    phiPreFiltering = cms.double(0.3),
    extraHitRZtolerance = cms.double(0.),
    fnSigmaRZ = cms.double(2.0),
    chi2VsPtCut = cms.bool(True),
    maxChi2 = cms.double(3.0),
    pt_interv = cms.vdouble(0.7,1.0,2.0,5.0),
    chi2_cuts = cms.vdouble(),
    refitHits = cms.bool(True),
    extraPhiKDBox = cms.double(0.),
    SimpleMagneticField = cms.string(''),
#    SimpleMagneticField = cms.string('ParabolicMf'),
    ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
    debug = cms.bool(False),
    detIdsToDebug = cms.vint32(0,0,0)
)
tobTecStepSeedsTripl.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
tobTecStepSeedsTripl.RegionFactoryPSet.RegionPSet.ptMin = 0.65
tobTecStepSeedsTripl.RegionFactoryPSet.RegionPSet.originHalfLength = 20.0
tobTecStepSeedsTripl.RegionFactoryPSet.RegionPSet.originRadius = 3.5
tobTecStepSeedsTripl.SeedCreatorPSet.OriginTransverseErrorMultiplier = 1.0
tobTecStepSeedsTripl.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(False),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
)
# PAIR SEEDING LAYERS
tobTecStepSeedLayersPair = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('TOB1+TEC1_pos','TOB1+TEC1_neg', 
                            'TEC1_pos+TEC2_pos','TEC1_neg+TEC2_neg', 
                            'TEC2_pos+TEC3_pos','TEC2_neg+TEC3_neg', 
                            'TEC3_pos+TEC4_pos','TEC3_neg+TEC4_neg', 
                            'TEC4_pos+TEC5_pos','TEC4_neg+TEC5_neg', 
                            'TEC5_pos+TEC6_pos','TEC5_neg+TEC6_neg', 
                            'TEC6_pos+TEC7_pos','TEC6_neg+TEC7_neg'),
    TOB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'),
         matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
         skipClusters   = cms.InputTag('tobTecStepClusters')
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        skipClusters = cms.InputTag('tobTecStepClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(5),
        maxRing = cms.int32(5)
    )
)
# PAIR SEEDS
import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff
tobTecStepSeedsPair = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone()
tobTecStepSeedsPair.OrderedHitsFactoryPSet.ComponentName = cms.string('StandardHitPairGenerator')
tobTecStepSeedsPair.OrderedHitsFactoryPSet.SeedingLayers = 'tobTecStepSeedLayersPair'
tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.ptMin = 0.8
tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.originHalfLength = 20.0
tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.originRadius = 3.5
tobTecStepSeedsPair.SeedCreatorPSet.OriginTransverseErrorMultiplier = 2.0
tobTecStepSeedsPair.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(False),
        FilterStripHits = cms.bool(True),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
)

import RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi
tobTecStepSeeds = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone()
tobTecStepSeeds.seedCollections = cms.VInputTag(cms.InputTag('tobTecStepSeedsTripl'),cms.InputTag('tobTecStepSeedsPair'))


TobTecStep = cms.Sequence(tobTecStepClusters*
                          tobTecStepSeedLayersTripl*
                          tobTecStepSeedsTripl*
                          tobTecStepSeedLayersPair*
                          tobTecStepSeedsPair*
                          tobTecStepSeeds*
                          tobTecStepTrackCandidates*
                          tobTecStepTracks*
                          tobTecStepSelector)
