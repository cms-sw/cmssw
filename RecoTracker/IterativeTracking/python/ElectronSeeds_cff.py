import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

from RecoLocalTracker.SubCollectionProducers.SeedClusterRemover_cfi import seedClusterRemover
initialStepSeedClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("initialStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("pixelLessStepClusters")
)
eras.trackingPhase1PU70.toModify(initialStepSeedClusterMask, oldClusterRemovalInfo = "highPtTripletStepClusters")
highPtTripletStepSeedClusterMask = seedClusterRemover.clone( # for Phase1PU70
    trajectories = "highPtTripletStepSeeds",
    oldClusterRemovalInfo = cms.InputTag("initialStepSeedClusterMask")
)
pixelPairStepSeedClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("pixelPairStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("initialStepSeedClusterMask")
)
eras.trackingPhase1PU70.toModify(pixelPairStepSeedClusterMask, oldClusterRemovalInfo = "highPtTripletStepSeedClusterMask")
# This is a pure guess to use lowPtTripletStep for phase1 here instead of the pixelPair in Run2 configuration
lowPtTripletStepSeedClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("lowPtTripletStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("initialStepSeedClusterMask")
)
mixedTripletStepSeedClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("mixedTripletStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("pixelPairStepSeedClusterMask")
)
eras.trackingPhase1.toModify(mixedTripletStepSeedClusterMask,
    oldClusterRemovalInfo = "lowPtTripletStepSeedClusterMask"
)
pixelLessStepSeedClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("pixelLessStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("mixedTripletStepSeedClusterMask")
)

tripletElectronSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2+BPix3', 
                            'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', 
                            'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
    HitProducer = cms.string('siPixelRecHits'),
    skipClusters = cms.InputTag('pixelLessStepSeedClusterMask')
    ),
    FPix = cms.PSet(
    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
    HitProducer = cms.string('siPixelRecHits'),
    skipClusters = cms.InputTag('pixelLessStepSeedClusterMask')
    )
)
_layerListForPhase1 = [
    'BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
    'BPix1+BPix3+BPix4', 'BPix1+BPix2+BPix4',
    'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
    'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
    'BPix1+BPix3+FPix1_pos', 'BPix1+BPix3+FPix1_neg',
    'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
    'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
    'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
    'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
    'BPix1+FPix1_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix3_neg'
]
eras.trackingPhase1.toModify(tripletElectronSeedLayers, layerList = _layerListForPhase1)
eras.trackingPhase1PU70.toModify(tripletElectronSeedLayers,
    layerList = _layerListForPhase1,
    BPix = dict(skipClusters = 'pixelPairStepSeedClusterMask'),
    FPix = dict(skipClusters = 'pixelPairStepSeedClusterMask')
)

import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
tripletElectronSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 1.0,
    originRadius = 0.02,
    nSigmaZ = 4.0
    )
    )
)
tripletElectronSeeds.OrderedHitsFactoryPSet.SeedingLayers = cms.InputTag('tripletElectronSeedLayers')
eras.trackingPhase1PU70.toModify(tripletElectronSeeds,
    OrderedHitsFactoryPSet = dict(maxElement = cms.uint32(0)), # not sure if this has any effect
)

from RecoLocalTracker.SubCollectionProducers.SeedClusterRemover_cfi import seedClusterRemover
tripletElectronClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("tripletElectronSeeds"),
    oldClusterRemovalInfo = cms.InputTag("pixelLessStepSeedClusterMask")
)

pixelPairElectronSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3', 
                            'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 
                            'BPix1+FPix2_pos', 'BPix1+FPix2_neg', 
                            'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 
                            'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
    TTRHBuilder = cms.string('WithTrackAngle'),
    HitProducer = cms.string('siPixelRecHits'),
    skipClusters = cms.InputTag('tripletElectronClusterMask')
    ),
    FPix = cms.PSet(
    TTRHBuilder = cms.string('WithTrackAngle'),
    HitProducer = cms.string('siPixelRecHits'),
    skipClusters = cms.InputTag('tripletElectronClusterMask')
    )
)
eras.trackingPhase1.toModify(pixelPairElectronSeedLayers,
    layerList = [
        'BPix1+BPix2', 'BPix1+BPix3', 'BPix1+BPix4',
        'BPix2+BPix3', 'BPix2+BPix4',
        'BPix3+BPix4',
        'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 
        'BPix1+FPix2_pos', 'BPix1+FPix2_neg', 
        'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 
        'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg',
        'FPix1_pos+FPix3_pos', 'FPix1_neg+FPix3_neg',
        'FPix2_pos+FPix3_pos', 'FPix2_neg+FPix3_neg' 
    ]
)

import RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff
pixelPairElectronSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff.globalSeedsFromPairsWithVertices.clone()
pixelPairElectronSeeds.RegionFactoryPSet.RegionPSet.ptMin = 1.0
pixelPairElectronSeeds.RegionFactoryPSet.RegionPSet.originRadius = 0.015
pixelPairElectronSeeds.RegionFactoryPSet.RegionPSet.fixedError = 0.03
pixelPairElectronSeeds.OrderedHitsFactoryPSet.SeedingLayers = cms.InputTag('pixelPairElectronSeedLayers')

stripPairElectronSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('TIB1+TIB2', 'TIB1+TID1_pos', 'TIB1+TID1_neg', 'TID2_pos+TID3_pos', 'TID2_neg+TID3_neg',
                            'TEC1_pos+TEC2_pos','TEC2_pos+TEC3_pos','TEC3_pos+TEC4_pos','TEC3_pos+TEC5_pos',
                            'TEC1_neg+TEC2_neg','TEC2_neg+TEC3_neg','TEC3_neg+TEC4_neg','TEC3_neg+TEC5_neg'),
    TIB = cms.PSet(
    TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone')),
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    skipClusters = cms.InputTag('tripletElectronClusterMask')
    ),
    TID = cms.PSet(
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    skipClusters = cms.InputTag('tripletElectronClusterMask'),
    useRingSlector = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone')),
    minRing = cms.int32(1),
    maxRing = cms.int32(2)
    ),
    TEC = cms.PSet(
    matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
    skipClusters = cms.InputTag('tripletElectronClusterMask'),
    useRingSlector = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'), clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone')),
    minRing = cms.int32(1),
    maxRing = cms.int32(2)
    )
)

import RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff
stripPairElectronSeeds = RecoTracker.TkSeedGenerator.GlobalMixedSeeds_cff.globalMixedSeeds.clone()
stripPairElectronSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'stripPairElectronSeedLayers'
stripPairElectronSeeds.RegionFactoryPSet.RegionPSet.ptMin = 1.0
stripPairElectronSeeds.RegionFactoryPSet.RegionPSet.originHalfLength = 12.0
stripPairElectronSeeds.RegionFactoryPSet.RegionPSet.originRadius = 0.4

###This seed collection is produced for electron reconstruction
import RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi
newCombinedSeeds = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone(
    seedCollections = cms.VInputTag(
      cms.InputTag('initialStepSeeds'),
      cms.InputTag('pixelPairStepSeeds'),
      cms.InputTag('mixedTripletStepSeeds'),
      cms.InputTag('pixelLessStepSeeds'),
      cms.InputTag('tripletElectronSeeds'),
      cms.InputTag('pixelPairElectronSeeds'),
      cms.InputTag('stripPairElectronSeeds')
      )
)
eras.trackingPhase1.toModify(newCombinedSeeds, seedCollections = [
    'initialStepSeeds',
    'highPtTripletStepSeeds',
    'mixedTripletStepSeeds',
    'pixelLessStepSeeds',
    'tripletElectronSeeds',
    'pixelPairElectronSeeds',
    'stripPairElectronSeeds'
])
eras.trackingPhase1PU70.toModify(newCombinedSeeds, seedCollections = [
    'initialStepSeeds',
    'highPtTripletStepSeeds',
    'pixelPairStepSeeds',
    'tripletElectronSeeds'
])

electronSeedsSeq = cms.Sequence( initialStepSeedClusterMask*
                                 pixelPairStepSeedClusterMask*
                                 mixedTripletStepSeedClusterMask*
                                 pixelLessStepSeedClusterMask*
                                 tripletElectronSeedLayers*
                                 tripletElectronSeeds*
                                 tripletElectronClusterMask*
                                 pixelPairElectronSeedLayers*
                                 pixelPairElectronSeeds*
                                 stripPairElectronSeedLayers*
                                 stripPairElectronSeeds*
                                 newCombinedSeeds)
_electronSeedsSeq_Phase1 = electronSeedsSeq.copy()
_electronSeedsSeq_Phase1.replace(pixelPairStepSeedClusterMask, lowPtTripletStepSeedClusterMask)
eras.trackingPhase1.toReplaceWith(electronSeedsSeq, _electronSeedsSeq_Phase1)
eras.trackingPhase1PU70.toReplaceWith(electronSeedsSeq, cms.Sequence(
    initialStepSeedClusterMask*
    highPtTripletStepSeedClusterMask*
    pixelPairStepSeedClusterMask*
    tripletElectronSeedLayers*
    tripletElectronSeeds*
    newCombinedSeeds
))
