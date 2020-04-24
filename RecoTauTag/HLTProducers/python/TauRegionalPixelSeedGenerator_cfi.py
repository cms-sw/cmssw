import FWCore.ParameterSet.Config as cms

tauRegionalPixelSeedGenerator = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.InputTag('PixelLayerPairs')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('TauRegionalPixelSeedGenerator'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            deltaPhiRegion = cms.double(0.1),
            originHalfLength = cms.double(0.2),
            originRadius = cms.double(0.2),
            deltaEtaRegion = cms.double(0.1),
            ptMin = cms.double(5.0),
            JetSrc = cms.InputTag("icone5Tau1"),
            originZPos = cms.double(0.0),
            vertexSrc = cms.InputTag("pixelVertices"),
            howToUseMeasurementTracker = cms.string("ForSiStrips"),
            measurementTrackerName = cms.InputTag("MeasurementTrackerEvent"),
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)


