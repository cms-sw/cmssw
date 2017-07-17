import FWCore.ParameterSet.Config as cms

HITmixedPairRegGenerator = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.InputTag('MixedLayerPairs')
    ),
    SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('none')
    ),
    RegionFactoryPSet = cms.PSet(
        ComponentName = cms.string('HITRegionalPixelSeedGenerator'),
        RegionPSet = cms.PSet(
            precise = cms.bool(True),
            deltaEtaTrackRegion = cms.double(0.3),
            deltaPhiTrackRegion = cms.double(0.3),
            isoTrackSrc = cms.InputTag("isolPixelTrackFilterL2"),
            deltaEtaL1JetRegion = cms.double(0.3),
            useTracks = cms.bool(False),
            originRadius = cms.double(0.2),
            trackSrc = cms.InputTag("pixelTracks"),
            useL1Jets = cms.bool(False),
            ptMin = cms.double(0.9),
            useIsoTracks = cms.bool(True),
            l1tjetSrc = cms.InputTag("l1extraParticles","Tau"),
            deltaPhiL1JetRegion = cms.double(0.3),
            vertexSrc = cms.InputTag("pixelVertices"),
            fixedReg = cms.bool(False),
            etaCenter = cms.double(0.0),
            phiCenter = cms.double(0.0),
            originZPos = cms.double(0.0),
            originHalfLength = cms.double(0.2)
        )
    ),
    TTRHBuilder = cms.string('WithTrackAngle')
)


