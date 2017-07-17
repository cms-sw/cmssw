import FWCore.ParameterSet.Config as cms

from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *
HITpixelTripletRegGenerator = cms.EDProducer("SeedGeneratorFromRegionHitsEDProducer",
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.InputTag('PixelLayerTriplets'),
        PixelTripletHLTGenerator = cms.PSet(
            useBending = cms.bool(True),
            useFixedPreFiltering = cms.bool(False),
            ComponentName = cms.string('PixelTripletHLTGenerator'),
            extraHitRPhitolerance = cms.double(0.032),
            useMultScattering = cms.bool(True),
            phiPreFiltering = cms.double(0.3),
            extraHitRZtolerance = cms.double(0.037)
        ),
        GeneratorPSet = cms.PSet(
            PixelTripletHLTGenerator
        )
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
            ptMin = cms.double(0.5),
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


