import FWCore.ParameterSet.Config as cms

TSGFromPixelPairsBlock = cms.PSet(
    SeedGeneratorParameters = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitPairGenerator'),
            SeedingLayers = cms.string('PixelLayerPairs')
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)

