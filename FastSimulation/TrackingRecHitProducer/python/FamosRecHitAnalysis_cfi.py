import FWCore.ParameterSet.Config as cms

FamosRecHitAnalysis = cms.EDFilter("FamosRecHitAnalysis",
    RecHits = cms.InputTag("trackerGSRecHitTranslator"),
    ROUList = cms.VInputTag(cms.InputTag("mix","famosSimHitsTrackerHits")),
    RootFileName = cms.string('FamosRecHitAnalysis.root'),
    # Switch between old and new parametrization
    UseCMSSWPixelParametrization = cms.bool(True),
    PixelMultiplicityFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelData.root'),
    PixelBarrelResolutionFileNew = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution38T.root'),
    PixelBarrelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution.root'),
    PixelMultiplicityFileNew = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelData38T.root'),
    PixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution.root'),
    PixelForwardResolutionFileNew = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution38T.root'),

    #Barrel
    AlphaBarrel_BinN = cms.int32(4),
    AlphaBarrel_BinNNew = cms.int32(4),
    AlphaBarrel_BinMin = cms.double(-0.2),
    AlphaBarrel_BinMinNew = cms.double(-0.2),
    AlphaBarrel_BinWidth = cms.double(0.1),
    AlphaBarrel_BinWidthNew = cms.double(0.1),
    AlphaBarrelMultiplicity = cms.int32(4),
    AlphaBarrelMultiplicityNew = cms.int32(4),
    BetaBarrel_BinN = cms.int32(7),
    BetaBarrel_BinNNew = cms.int32(7),
    BetaBarrel_BinMin = cms.double(0.0),
    BetaBarrel_BinMinNew = cms.double(0.0),
    BetaBarrel_BinWidth = cms.double(0.2),
    BetaBarrel_BinWidthNew = cms.double(0.2),
    BetaBarrelMultiplicity = cms.int32(6),
    BetaBarrelMultiplicityNew = cms.int32(7),

    # Forward
    AlphaForward_BinN = cms.int32(0),
    AlphaForward_BinNNew = cms.int32(0),
    AlphaForward_BinMin = cms.double(0.0),
    AlphaForward_BinMinNew = cms.double(0.0),
    AlphaForward_BinWidth = cms.double(0.0),
    AlphaForward_BinWidthNew = cms.double(0.0),
    AlphaForwardMultiplicity = cms.int32(3),
    AlphaForwardMultiplicityNew = cms.int32(3),
    BetaForward_BinN = cms.int32(0),
    BetaForward_BinNNew = cms.int32(0),
    BetaForward_BinMin = cms.double(0.0),
    BetaForward_BinMinNew = cms.double(0.0),
    BetaForward_BinWidth = cms.double(0.0),
    BetaForward_BinWidthNew = cms.double(0.0),
    BetaForwardMultiplicityNew = cms.int32(3),
    BetaForwardMultiplicity = cms.int32(3)
)



