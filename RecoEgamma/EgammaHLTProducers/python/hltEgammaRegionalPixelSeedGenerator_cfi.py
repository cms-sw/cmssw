import FWCore.ParameterSet.Config as cms

hltEgammaRegionalPixelSeedGenerator = cms.EDProducer("EgammaHLTRegionalPixelSeedGeneratorProducers",
    deltaPhiRegion = cms.double(0.3), ## .177 

    vertexZ = cms.double(0.0),
    originHalfLength = cms.double(15.0),
    BSProducer = cms.InputTag("offlineBeamSpot"),
    UseZInVertex = cms.bool(False),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitPairGenerator'),
        SeedingLayers = cms.string('PixelLayerPairs')
    ),
    deltaEtaRegion = cms.double(0.3), ## .177 

    #    string HitProducer = "siPixelRecHits"
    ptMin = cms.double(1.5),
    candTag = cms.InputTag("hltRecoEcalCandidate"),
    TTRHBuilder = cms.string('WithTrackAngle'),
    candTagEle = cms.InputTag("pixelMatchElectrons"),
    originRadius = cms.double(0.02) ##0.2

)


