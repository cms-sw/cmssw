import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.PixelLessStep_cff import *

# SEEDING LAYERS
pixelLessStepSeedLayers = cms.ESProducer("SeedingLayersESProducer",
    ComponentName = cms.string('pixelLessStepSeedLayers'),
    layerList = cms.vstring(
    #TIB
    'TIB1+TIB2+MTIB3',
    #TIB+TID
    'TIB1+TIB2+MTID1_pos','TIB1+TIB2+MTID1_neg',
    #TID
    'TID1_pos+TID2_pos+TID3_pos','TID1_neg+TID2_neg+TID3_neg',#ring 1-2 (matched)
    'TID1_pos+TID2_pos+MTID3_pos','TID1_neg+TID2_neg+MTID3_neg',#ring 3 (mono)
    'TID1_pos+TID2_pos+MTEC1_pos','TID1_neg+TID2_neg+MTEC1_neg',
    #TID+TEC RING 1-3
    'TID2_pos+TID3_pos+TEC1_pos','TID2_neg+TID3_neg+TEC1_neg',#ring 1-2 (matched)
    'TID2_pos+TID3_pos+MTEC1_pos','TID2_neg+TID3_neg+MTEC1_neg',#ring 3 (mono)
    #TEC RING 1-3
    'TEC1_pos+TEC2_pos+TEC3_pos', 'TEC1_neg+TEC2_neg+TEC3_neg',
    'TEC1_pos+TEC2_pos+MTEC3_pos','TEC1_neg+TEC2_neg+MTEC3_neg',
    'TEC1_pos+TEC2_pos+TEC4_pos', 'TEC1_neg+TEC2_neg+TEC4_neg',
    'TEC1_pos+TEC2_pos+MTEC4_pos','TEC1_neg+TEC2_neg+MTEC4_neg',
    'TEC2_pos+TEC3_pos+TEC4_pos', 'TEC2_neg+TEC3_neg+TEC4_neg',
    'TEC2_pos+TEC3_pos+MTEC4_pos','TEC2_neg+TEC3_neg+MTEC4_neg',
    'TEC2_pos+TEC3_pos+TEC5_pos', 'TEC2_neg+TEC3_neg+TEC5_neg',
    'TEC2_pos+TEC3_pos+TEC6_pos', 'TEC2_neg+TEC3_neg+TEC6_neg',
    'TEC3_pos+TEC4_pos+TEC5_pos', 'TEC3_neg+TEC4_neg+TEC5_neg',
    'TEC3_pos+TEC4_pos+MTEC5_pos','TEC3_neg+TEC4_neg+MTEC5_neg',
    'TEC3_pos+TEC5_pos+TEC6_pos', 'TEC3_neg+TEC5_neg+TEC6_neg',
    'TEC4_pos+TEC5_pos+TEC6_pos', 'TEC4_neg+TEC5_neg+TEC6_neg'    
    ),
    TIB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'),
         matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
         skipClusters   = cms.InputTag('pixelLessStepSeedClusters')
    ),
    MTIB = cms.PSet(
         TTRHBuilder    = cms.string('WithTrackAngle'),
         skipClusters   = cms.InputTag('pixelLessStepSeedClusters'),
         rphiRecHits    = cms.InputTag("siStripMatchedRecHits","rphiRecHit")
    ),
    TID = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        skipClusters = cms.InputTag('pixelLessStepSeedClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    MTID = cms.PSet(
        rphiRecHits    = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        skipClusters = cms.InputTag('pixelLessStepSeedClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(3),
        maxRing = cms.int32(3)
    ),
    TEC = cms.PSet(
        matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
        skipClusters = cms.InputTag('pixelLessStepSeedClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(1),
        maxRing = cms.int32(2)
    ),
    MTEC = cms.PSet(
        rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        skipClusters = cms.InputTag('pixelLessStepSeedClusters'),
        useRingSlector = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        minRing = cms.int32(3),
        maxRing = cms.int32(3)
    )
)

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
pixelLessStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone()
#OrderedHitsFactory
pixelLessStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'pixelLessStepSeedLayers'
pixelLessStepSeeds.OrderedHitsFactoryPSet.ComponentName = 'StandardMultiHitGenerator'
import RecoTracker.TkSeedGenerator.MultiHitGeneratorFromChi2_cfi
pixelLessStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet = RecoTracker.TkSeedGenerator.MultiHitGeneratorFromChi2_cfi.MultiHitGeneratorFromChi2.clone()
#SeedCreator
pixelLessStepSeeds.SeedCreatorPSet.ComponentName = 'SeedFromConsecutiveHitsTripletOnlyCreator'
#RegionFactory
pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.4
pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 12.0 #15 def, 12 pls1
pixelLessStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 1.0 #2.5 def, 1.0 pls1

#SeedComparitor
pixelLessStepSeeds.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(False),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter')
)

PixelLessStep = cms.Sequence(pixelLessStepClusters*
                             pixelLessStepSeedClusters*
                             pixelLessStepSeeds*
                             pixelLessStepTrackCandidates*
                             pixelLessStepTracks*
                             pixelLessStepSelector*
                             pixelLessStep)
