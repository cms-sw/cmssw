import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SubCollectionProducers.SeedClusterRemover_cfi import seedClusterRemover
initialStepSeedClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("initialStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("pixelLessStepClusters")
)
from Configuration.Eras.Modifier_trackingPhase1PU70_cff import trackingPhase1PU70
trackingPhase1PU70.toModify(initialStepSeedClusterMask, oldClusterRemovalInfo = "highPtTripletStepClusters")

from RecoLocalTracker.SubCollectionProducers.seedClusterRemoverPhase2_cfi import seedClusterRemoverPhase2
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
trackingPhase2PU140.toReplaceWith(initialStepSeedClusterMask, seedClusterRemoverPhase2.clone(
    trajectories = cms.InputTag("initialStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("highPtTripletStepClusters")
    )
)

highPtTripletStepSeedClusterMask = seedClusterRemover.clone( # for Phase1PU70
    trajectories = "highPtTripletStepSeeds",
    oldClusterRemovalInfo = cms.InputTag("initialStepSeedClusterMask")
)
pixelPairStepSeedClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("pixelPairStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("initialStepSeedClusterMask")
)
trackingPhase1PU70.toModify(pixelPairStepSeedClusterMask, oldClusterRemovalInfo = "highPtTripletStepSeedClusterMask")

trackingPhase2PU140.toReplaceWith(highPtTripletStepSeedClusterMask, seedClusterRemoverPhase2.clone(
    trajectories = cms.InputTag("highPtTripletStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("initialStepSeedClusterMask")
    )
)
trackingPhase2PU140.toReplaceWith(pixelPairStepSeedClusterMask, seedClusterRemoverPhase2.clone(
    trajectories = cms.InputTag("detachedQuadStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("highPtTripletStepSeedClusterMask")
    )
)

# This is a pure guess to use detachedTripletStep for phase1 here instead of the pixelPair in Run2 configuration
detachedTripletStepSeedClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("lowPtTripletStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("initialStepSeedClusterMask")
)
mixedTripletStepSeedClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("mixedTripletStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("pixelPairStepSeedClusterMask")
)
from Configuration.Eras.Modifier_trackingPhase1_cff import trackingPhase1
trackingPhase1.toModify(mixedTripletStepSeedClusterMask,
    oldClusterRemovalInfo = "detachedTripletStepSeedClusterMask"
)
from Configuration.Eras.Modifier_trackingPhase1QuadProp_cff import trackingPhase1QuadProp
trackingPhase1QuadProp.toModify(mixedTripletStepSeedClusterMask,
    oldClusterRemovalInfo = "detachedTripletStepSeedClusterMask"
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
trackingPhase1.toModify(tripletElectronSeedLayers, layerList = _layerListForPhase1)
trackingPhase1QuadProp.toModify(tripletElectronSeedLayers, layerList = _layerListForPhase1)
trackingPhase1PU70.toModify(tripletElectronSeedLayers,
    layerList = _layerListForPhase1,
    BPix = dict(skipClusters = 'pixelPairStepSeedClusterMask'),
    FPix = dict(skipClusters = 'pixelPairStepSeedClusterMask')
)
trackingPhase2PU140.toModify(tripletElectronSeedLayers,
    layerList = _layerListForPhase1,
    BPix = dict(skipClusters = 'pixelPairStepSeedClusterMask'),
    FPix = dict(skipClusters = 'pixelPairStepSeedClusterMask')
)

from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpot_cfi import globalTrackingRegionFromBeamSpot as _globalTrackingRegionFromBeamSpot
tripletElectronTrackingRegions = _globalTrackingRegionFromBeamSpot.clone(RegionPSet = dict(
    ptMin = 1.0,
    originRadius = 0.02,
    nSigmaZ = 4.0
))
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
tripletElectronHitDoublets = _hitPairEDProducer.clone(
    seedingLayers = "tripletElectronSeedLayers",
    trackingRegions = "tripletElectronTrackingRegions",
    maxElement = 0,
    produceIntermediateHitDoublets = True,
)
from RecoPixelVertexing.PixelTriplets.pixelTripletHLTEDProducer_cfi import pixelTripletHLTEDProducer as _pixelTripletHLTEDProducer
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
tripletElectronHitTriplets = _pixelTripletHLTEDProducer.clone(
    doublets = "tripletElectronHitDoublets",
    maxElement = 1000000,
    produceSeedingHitSets = True,
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
tripletElectronSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = "tripletElectronHitTriplets",
)
trackingPhase1PU70.toModify(tripletElectronHitTriplets,
    maxElement = 0, # not sure if this has any effect
)
trackingPhase2PU140.toModify(tripletElectronHitTriplets,
    maxElement = 0,
)

from RecoLocalTracker.SubCollectionProducers.SeedClusterRemover_cfi import seedClusterRemover
tripletElectronClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("tripletElectronSeeds"),
    oldClusterRemovalInfo = cms.InputTag("pixelLessStepSeedClusterMask")
)
trackingPhase2PU140.toReplaceWith(tripletElectronClusterMask, seedClusterRemoverPhase2.clone(
    trajectories = cms.InputTag("tripletElectronSeeds"),
    oldClusterRemovalInfo = cms.InputTag("pixelLessStepSeedClusterMask")
    )
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
_layerListForPhase1 = [
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
trackingPhase1.toModify(pixelPairElectronSeedLayers, layerList = _layerListForPhase1)
trackingPhase1QuadProp.toModify(pixelPairElectronSeedLayers, layerList = _layerListForPhase1)

from RecoTracker.TkTrackingRegions.globalTrackingRegionWithVertices_cff import globalTrackingRegionWithVertices as _globalTrackingRegionWithVertices
pixelPairElectronTrackingRegions = _globalTrackingRegionWithVertices.clone(RegionPSet = dict(
    ptMin = 1.0,
    originRadius = 0.015,
    fixedError = 0.03,
))
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
pixelPairElectronHitDoublets = _hitPairEDProducer.clone(
    seedingLayers = "pixelPairElectronSeedLayers",
    trackingRegions = "pixelPairElectronTrackingRegions",
    maxElement = 1000000,
    produceSeedingHitSets = True,
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
pixelPairElectronSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = "pixelPairElectronHitDoublets",
)

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

from RecoTracker.TkTrackingRegions.globalTrackingRegionFromBeamSpotFixedZ_cfi import globalTrackingRegionFromBeamSpotFixedZ as _globalTrackingRegionFromBeamSpotFixedZ
stripPairElectronTrackingRegions = _globalTrackingRegionFromBeamSpotFixedZ.clone(RegionPSet = dict(
    ptMin = 1.0,
    originHalfLength = 12.0,
    originRadius = 0.4,
))
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import hitPairEDProducer as _hitPairEDProducer
stripPairElectronHitDoublets = _hitPairEDProducer.clone(
    seedingLayers = "stripPairElectronSeedLayers",
    trackingRegions = "stripPairElectronTrackingRegions",
    maxElement = 1000000,
    produceSeedingHitSets = True,
)
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cff import seedCreatorFromRegionConsecutiveHitsEDProducer as _seedCreatorFromRegionConsecutiveHitsEDProducer
stripPairElectronSeeds = _seedCreatorFromRegionConsecutiveHitsEDProducer.clone(
    seedingHitSets = "stripPairElectronHitDoublets",
)


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
_seedCollections_Phase1 = [
    'initialStepSeeds',
    'highPtTripletStepSeeds',
    'mixedTripletStepSeeds',
    'pixelLessStepSeeds',
    'tripletElectronSeeds',
    'pixelPairElectronSeeds',
    'stripPairElectronSeeds'
]
trackingPhase1.toModify(newCombinedSeeds, seedCollections = _seedCollections_Phase1)
trackingPhase1QuadProp.toModify(newCombinedSeeds, seedCollections = _seedCollections_Phase1)
trackingPhase1PU70.toModify(newCombinedSeeds, seedCollections = [
    'initialStepSeeds',
    'highPtTripletStepSeeds',
    'pixelPairStepSeeds',
    'tripletElectronSeeds'
])
trackingPhase2PU140.toModify(newCombinedSeeds, seedCollections = [
    'initialStepSeeds',
    'highPtTripletStepSeeds',
    'tripletElectronSeeds'
])

electronSeedsSeq = cms.Sequence( initialStepSeedClusterMask*
                                 pixelPairStepSeedClusterMask*
                                 mixedTripletStepSeedClusterMask*
                                 pixelLessStepSeedClusterMask*
                                 tripletElectronSeedLayers*
                                 tripletElectronTrackingRegions*
                                 tripletElectronHitDoublets*
                                 tripletElectronHitTriplets*
                                 tripletElectronSeeds*
                                 tripletElectronClusterMask*
                                 pixelPairElectronSeedLayers*
                                 pixelPairElectronTrackingRegions*
                                 pixelPairElectronHitDoublets*
                                 pixelPairElectronSeeds*
                                 stripPairElectronSeedLayers*
                                 stripPairElectronTrackingRegions*
                                 stripPairElectronHitDoublets*
                                 stripPairElectronSeeds*
                                 newCombinedSeeds)
_electronSeedsSeq_Phase1 = electronSeedsSeq.copy()
_electronSeedsSeq_Phase1.replace(pixelPairStepSeedClusterMask, detachedTripletStepSeedClusterMask)
trackingPhase1.toReplaceWith(electronSeedsSeq, _electronSeedsSeq_Phase1)
trackingPhase1QuadProp.toReplaceWith(electronSeedsSeq, _electronSeedsSeq_Phase1)
trackingPhase1PU70.toReplaceWith(electronSeedsSeq, cms.Sequence(
    initialStepSeedClusterMask*
    highPtTripletStepSeedClusterMask*
    pixelPairStepSeedClusterMask*
    tripletElectronSeedLayers*
    tripletElectronTrackingRegions*
    tripletElectronHitDoublets*
    tripletElectronHitTriplets*
    tripletElectronSeeds*
    newCombinedSeeds
))
trackingPhase2PU140.toReplaceWith(electronSeedsSeq, cms.Sequence(
    initialStepSeedClusterMask*
    highPtTripletStepSeedClusterMask*
    pixelPairStepSeedClusterMask*
    tripletElectronSeedLayers*
    tripletElectronTrackingRegions*
    tripletElectronHitDoublets*
    tripletElectronHitTriplets*
    tripletElectronSeeds*
    newCombinedSeeds
))

