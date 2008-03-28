import FWCore.ParameterSet.Config as cms

seedsFromL1Muon = cms.EDFilter("TSGFromL1Muon",
    FitterPSet = cms.PSet(
        ComponentName = cms.string('PixelFitterByHelixProjections'),
        TTRHBuilder = cms.string('PixelTTRHBuilderWithoutAngleSeedsFromL1Muon')
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('PixelLayerPairs')
    )
)


