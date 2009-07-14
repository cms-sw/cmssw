import FWCore.ParameterSet.Config as cms

FamosRecHitAnalysis = cms.EDFilter("FamosRecHitAnalysis",
    AlphaBarrel_BinN = cms.int32(4),
    AlphaBarrelMultiplicityNew = cms.int32(4),
    RecHits = cms.InputTag("trackerGSRecHitTranslator"),
    BetaForwardMultiplicityNew = cms.int32(3),
    BetaForward_BinMinNew = cms.double(0.0),
    AlphaBarrel_BinWidth = cms.double(0.1),
    BetaBarrelMultiplicityNew = cms.int32(7),
    AlphaForward_BinN = cms.int32(0),
    BetaBarrel_BinMin = cms.double(0.0),
    BetaBarrel_BinWidthNew = cms.double(0.2),
    # Pixel
    PixelMultiplicityFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelData.root'),
    BetaForward_BinNNew = cms.int32(0),
    BetaBarrel_BinWidth = cms.double(0.2),
    # Switch between old and new parametrization
    UseCMSSWPixelParametrization = cms.bool(True),
    BetaBarrel_BinN = cms.int32(7),
    AlphaForward_BinNNew = cms.int32(0),
    PixelBarrelResolutionFileNew = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolutionNew.root'),
    AlphaForwardMultiplicity = cms.int32(3),
    AlphaForward_BinMin = cms.double(0.0),
    BetaForward_BinMin = cms.double(0.0),
    ROUList = cms.VInputTag(cms.InputTag("mix","famosSimHitsTrackerHits")),
    AlphaForward_BinMinNew = cms.double(0.0),
    PixelBarrelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution.root'),
    AlphaBarrel_BinMin = cms.double(-0.2),
    # Pixel CMSSW parameterization
    PixelMultiplicityFileNew = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelDataNew.root'),
    BetaForward_BinN = cms.int32(0),
    PixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution.root'),
    PixelForwardResolutionFileNew = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolutionNew.root'),
    BetaBarrel_BinMinNew = cms.double(0.0),
    AlphaBarrel_BinWidthNew = cms.double(0.1),
    BetaForward_BinWidth = cms.double(0.0),
    BetaForwardMultiplicity = cms.int32(3),
    BetaBarrelMultiplicity = cms.int32(6),
    AlphaBarrelMultiplicity = cms.int32(4),
    AlphaForward_BinWidthNew = cms.double(0.0),
    AlphaForward_BinWidth = cms.double(0.0),
    BetaBarrel_BinNNew = cms.int32(7),
    RootFileName = cms.string('FamosRecHitAnalysis.root'),
    BetaForward_BinWidthNew = cms.double(0.0),
    AlphaBarrel_BinMinNew = cms.double(-0.2),
    AlphaBarrel_BinNNew = cms.int32(4),
    AlphaForwardMultiplicityNew = cms.int32(3)
)



