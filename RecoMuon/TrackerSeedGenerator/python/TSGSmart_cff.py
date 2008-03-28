import FWCore.ParameterSet.Config as cms

SeedGeneratorParameters = cms.PSet(
    EtaBound = cms.double(2.0),
    ComponentName = cms.string('TSGSmart'),
    PixelPairGeneratorSet = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitPairGenerator'),
            SeedingLayers = cms.string('PixelLayerPairs')
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    PixelTripletGeneratorSet = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitTripletGenerator'),
            SeedingLayers = cms.string('PixelLayerTriplets'),
            GeneratorPSet = cms.PSet(
                useBending = cms.bool(True),
                useFixedPreFiltering = cms.bool(False),
                ComponentName = cms.string('PixelTripletHLTGenerator'),
                extraHitRPhitolerance = cms.double(0.06),
                useMultScattering = cms.bool(True),
                phiPreFiltering = cms.double(0.3),
                extraHitRZtolerance = cms.double(0.06)
            )
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    ),
    MixedGeneratorSet = cms.PSet(
        ComponentName = cms.string('TSGFromOrderedHits'),
        OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string('StandardHitPairGenerator'),
            SeedingLayers = cms.string('MixedLayerPairs')
        ),
        TTRHBuilder = cms.string('WithTrackAngle')
    )
)

