import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SubCollectionProducers.SeedClusterRemover_cfi import seedClusterRemover
initialStepSeedClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("initialStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("highPtTripletStepClusters")
)
highPtTripletStepSeedClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("highPtTripletStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("initialStepSeedClusterMask")
)
pixelPairStepSeedClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("pixelPairStepSeeds"),
    oldClusterRemovalInfo = cms.InputTag("highPtTripletStepSeedClusterMask")
)

tripletElectronSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
                            'BPix1+BPix3+BPix4', 'BPix1+BPix2+BPix4',
                            'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
                            'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
                            'BPix1+BPix3+FPix1_pos', 'BPix1+BPix3+FPix1_neg',
                            'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
                            'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
                            'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg',
                            'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
                            'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
                            'BPix1+FPix1_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix3_neg'),
    BPix = cms.PSet(
    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
    HitProducer = cms.string('siPixelRecHits'),
    skipClusters = cms.InputTag('pixelPairStepSeedClusterMask')
    ),
    FPix = cms.PSet(
    TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelTriplets'),
    HitProducer = cms.string('siPixelRecHits'),
    skipClusters = cms.InputTag('pixelPairStepSeedClusterMask')
    )
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
tripletElectronSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
tripletElectronSeeds.OrderedHitsFactoryPSet.maxElement = cms.uint32(0)

from RecoLocalTracker.SubCollectionProducers.SeedClusterRemover_cfi import seedClusterRemover
tripletElectronClusterMask = seedClusterRemover.clone(
    trajectories = cms.InputTag("tripletElectronSeeds"),
    oldClusterRemovalInfo = cms.InputTag("pixelLessStepSeedClusterMask")
)


###This seed collection is produced for electron reconstruction
import RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi
newCombinedSeeds = RecoTracker.TkSeedGenerator.GlobalCombinedSeeds_cfi.globalCombinedSeeds.clone(
    seedCollections = cms.VInputTag(
      cms.InputTag('initialStepSeeds'),
      cms.InputTag('highPtTripletStepSeeds'),
      cms.InputTag('pixelPairStepSeeds'),
      cms.InputTag('tripletElectronSeeds')
      )
)

electronSeedsSeq = cms.Sequence(initialStepSeedClusterMask*
                                highPtTripletStepSeedClusterMask*
                                pixelPairStepSeedClusterMask*
                                tripletElectronSeedLayers*
                                tripletElectronSeeds*
                                newCombinedSeeds)
