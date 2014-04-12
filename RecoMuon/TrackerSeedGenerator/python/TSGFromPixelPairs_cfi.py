import FWCore.ParameterSet.Config as cms

TSGFromPixelPairsBlock = cms.PSet(
    SeedGeneratorParameters = cms.PSet(
    ComponentName = cms.string('TSGFromOrderedHits'),
    OrderedHitsFactoryPSet = cms.PSet(
    ComponentName = cms.string('StandardHitPairGenerator'),
    SeedingLayers = cms.InputTag('PixelLayerPairs'),
    useOnDemandTracker = cms.untracked.int32( 0 ),
    maxElement = cms.uint32( 0 )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
    )
    )
