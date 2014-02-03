import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.TobTecStep_cff import *

# TRIPLET SEEDING LAYERS
tobTecStepSeedLayersTripl = cms.ESProducer("SeedingLayersESProducer",
    ComponentName = cms.string('tobTecStepSeedLayersTripl'),
    layerList = cms.vstring(
    #TOB
    'TOB1+TOB2+MTOB3',
    #TOB+MTEC
    'TOB1+TOB2+MTEC1_pos','TOB1+TOB2+MTEC1_neg',
    ),
    TOB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'),
         matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
         skipClusters   = cms.InputTag('tobTecStepSeedClusters')
    ),
    MTOB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'),
         skipClusters   = cms.InputTag('tobTecStepSeedClusters'),
         rphiRecHits    = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    MTEC = cms.PSet(
        rphiRecHits    = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        skipClusters = cms.InputTag('tobTecStepSeedClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(6),
        maxRing = cms.int32(7)
    )
)
# TRIPLET SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
tobTecStepSeedsTripl = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
#OrderedHitsFactory
tobTecStepSeedsTripl.OrderedHitsFactoryPSet.SeedingLayers = 'tobTecStepSeedLayersTripl'
tobTecStepSeedsTripl.OrderedHitsFactoryPSet.ComponentName = 'StandardMultiHitGenerator'
import RecoTracker.TkSeedGenerator.MultiHitGeneratorFromChi2_cfi
tobTecStepSeedsTripl.OrderedHitsFactoryPSet.GeneratorPSet = RecoTracker.TkSeedGenerator.MultiHitGeneratorFromChi2_cfi.MultiHitGeneratorFromChi2.clone(
    extraPhiKDBox = 0.01
    )
#RegionFactory
tobTecStepSeedsTripl.RegionFactoryPSet.RegionPSet.ptMin = 0.55
tobTecStepSeedsTripl.RegionFactoryPSet.RegionPSet.originHalfLength = 20.0
tobTecStepSeedsTripl.RegionFactoryPSet.RegionPSet.originRadius = 3.5
#SeedCreator
tobTecStepSeedsTripl.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsCreator' #empirically better than 'SeedFromConsecutiveHitsTripletOnlyCreator'
tobTecStepSeedsTripl.SeedCreatorPSet.OriginTransverseErrorMultiplier = 1.0
#SeedComparitor
tobTecStepSeedsTripl.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(False),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
)
# PAIR SEEDING LAYERS
tobTecStepSeedLayersPair = cms.ESProducer("SeedingLayersESProducer",
    ComponentName = cms.string('tobTecStepSeedLayersPair'),
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
         skipClusters   = cms.InputTag('tobTecStepSeedClusters')
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        skipClusters = cms.InputTag('tobTecStepSeedClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(5),
        maxRing = cms.int32(5)
    )
)
# PAIR SEEDS
import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff
tobTecStepSeedsPair = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone()
#OrderedHitsFactory
tobTecStepSeedsPair.OrderedHitsFactoryPSet.ComponentName = cms.string('StandardHitPairGenerator')
tobTecStepSeedsPair.OrderedHitsFactoryPSet.SeedingLayers = 'tobTecStepSeedLayersPair'
#RegionFactory
tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.ptMin = 0.6
tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.originHalfLength = 30.0
tobTecStepSeedsPair.RegionFactoryPSet.RegionPSet.originRadius = 6.0
#SeedCreator
tobTecStepSeedsPair.SeedCreatorPSet.OriginTransverseErrorMultiplier = 1.0
#SeedComparitor
tobTecStepSeedsPair.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(False),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
)

import RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi
tobTecStepSeeds = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone()
tobTecStepSeeds.seedCollections = cms.VInputTag(cms.InputTag('tobTecStepSeedsTripl'),cms.InputTag('tobTecStepSeedsPair'))


TobTecStep = cms.Sequence(tobTecStepClusters*
                          tobTecStepSeedClusters*
                          tobTecStepSeedsTripl*
                          tobTecStepSeedsPair*
                          tobTecStepSeeds*
                          tobTecStepTrackCandidates*
                          tobTecStepTracks*
                          tobTecStepSelector)
