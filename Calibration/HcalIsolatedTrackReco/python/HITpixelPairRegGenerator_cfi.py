import FWCore.ParameterSet.Config as cms

HITpixelPairRegGenerator = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.InputTag('PixelLayerPairs')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('HITRegionalPixelSeedGenerator'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            deltaEtaTrackRegion = cms.double(0.05),
            deltaPhiTrackRegion = cms.double(0.05),
            originHalfLength = cms.double(0.2),
            deltaEtaL1JetRegion = cms.double(0.3),
            useTracks = cms.bool(False),
            originRadius = cms.double(0.2),
            trackSrc = cms.InputTag("pixelTracks"),
            useL1Jets = cms.bool(False),
            useIsoTracks = cms.bool(True),
            l1tjetSrc = cms.InputTag("l1extraParticles","Tau"),
            deltaPhiL1JetRegion = cms.double(0.3),
            ptMin = cms.double(5.0),
            fixedReg = cms.bool(False),
            etaCenter = cms.double(0.0),
            phiCenter = cms.double(0.0),
            originZPos = cms.double(0.0),
            vertexSrc = cms.InputTag("pixelVertices")
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)


