import FWCore.ParameterSet.Config as cms

siTrackerGaussianSmearingRecHits = cms.EDProducer("SiTrackerGaussianSmearingRecHitConverter",

#    ROUList = cms.VInputTag(cms.InputTag("mix","famosSimHitsTrackerHits")),
    InputSimHits = cms.InputTag("famosSimHits","TrackerHits"),
    VerboseLevel = cms.untracked.int32(2),
    UseSigma = cms.bool(True),
    # matching of 1dim hits in double-sided modules
    # creating 2dim hits
    doRecHitMatching = cms.bool(True),
    # If you want to have RecHits == PSimHits (tracking with PSimHits)
    trackingPSimHits = cms.bool(False),

    # Set to (True) for taking the existence of dead modules into account:
    killDeadChannels = cms.bool(True),
    #
    DeltaRaysMomentumCut = cms.double(0.5),

    # Pixel
    AlphaBarrelMultiplicity = cms.int32(4),
    AlphaBarrel_BinWidthNew = cms.double(0.1),
    AlphaBarrel_BinN = cms.int32(4),
    AlphaBarrel_BinMinNew = cms.double(-0.2),
    AlphaBarrel_BinMin = cms.double(-0.2),
    AlphaBarrelMultiplicityNew = cms.int32(4),
    AlphaBarrel_BinWidth = cms.double(0.1),
    AlphaBarrel_BinNNew = cms.int32(4),
    AlphaForward_BinN = cms.int32(0),
    AlphaForward_BinMinNew = cms.double(0.0),
    AlphaForward_BinNNew = cms.int32(0),
    AlphaForward_BinWidthNew = cms.double(0.0),
    AlphaForwardMultiplicity = cms.int32(3),
    AlphaForwardMultiplicityNew = cms.int32(3),
    AlphaForward_BinWidth = cms.double(0.0),
    AlphaForward_BinMin = cms.double(0.0),
    BetaBarrel_BinNNew = cms.int32(7),
    BetaBarrelMultiplicity = cms.int32(6),
    BetaBarrel_BinWidthNew = cms.double(0.2),
    BetaBarrel_BinN = cms.int32(7),
    BetaBarrelMultiplicityNew = cms.int32(7),
    BetaBarrel_BinMin = cms.double(0.0),
    BetaBarrel_BinWidth = cms.double(0.2),
    BetaBarrel_BinMinNew = cms.double(0.0),
    BetaForward_BinMin = cms.double(0.0),
    BetaForwardMultiplicity = cms.int32(3),
    BetaForward_BinWidthNew = cms.double(0.0),
    BetaForward_BinWidth = cms.double(0.0),
    BetaForward_BinN = cms.int32(0),
    BetaForward_BinNNew = cms.int32(0),
    BetaForwardMultiplicityNew = cms.int32(3),
    BetaForward_BinMinNew = cms.double(0.0),

    # Needed to compute Pixel Errors
    PixelErrorParametrization = cms.string('NOTcmsim'),

    # Switch between old and new parametrization
    UseCMSSWPixelParametrization = cms.bool(True),

    # Pixel CMSSW Parametrization
    templateIdBarrel = cms.int32( 40 ),
    templateIdForward  = cms.int32( 41 ),
    PixelMultiplicityFile40T = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelData40T.root'),
    PixelMultiplicityFile38T = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelData38T.root'),
    PixelForwardResolutionFile40T = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution40T.root'),
    PixelForwardResolutionFile38T = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution38T.root'),
    PixelMultiplicityFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelData.root'),
    PixelBarrelResolutionFile40T = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution40T.root'),
    PixelBarrelResolutionFile38T = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution38T.root'),
    PixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution.root'),
    PixelBarrelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution.root'),
    NewPixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionForward38T.root'),
    NewPixelBarrelResolutionFile1 = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionBarrel38T.root'),
    NewPixelBarrelResolutionFile2 = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionBarrelEdge38T.root'),

    UseNewParametrization = cms.bool(True),

                                                  
    #converting energy loss from GeV to ADC counts
    GevPerElectron = cms.double(3.61e-09),
    ElectronsPerADC = cms.double(250.0),
             

    # Hit Finding Probabilities
    HitFindingProbability_PXB = cms.double(1.0),
    HitFindingProbability_PXF = cms.double(1.0),
    HitFindingProbability_TIB1 = cms.double(1.0),
    HitFindingProbability_TIB2 = cms.double(1.0),
    HitFindingProbability_TIB3 = cms.double(1.0),
    HitFindingProbability_TIB4 = cms.double(1.0),
    HitFindingProbability_TID1 = cms.double(1.0),
    HitFindingProbability_TID2 = cms.double(1.0),
    HitFindingProbability_TID3 = cms.double(1.0),
    HitFindingProbability_TOB1 = cms.double(1.0),
    HitFindingProbability_TOB2 = cms.double(1.0),
    HitFindingProbability_TOB3 = cms.double(1.0),
    HitFindingProbability_TOB5 = cms.double(1.0),
    HitFindingProbability_TOB4 = cms.double(1.0),
    HitFindingProbability_TOB6 = cms.double(1.0),
    HitFindingProbability_TEC1 = cms.double(1.0),
    HitFindingProbability_TEC2 = cms.double(1.0),
    HitFindingProbability_TEC3 = cms.double(1.0),
    HitFindingProbability_TEC4 = cms.double(1.0),
    HitFindingProbability_TEC5 = cms.double(1.0),
    HitFindingProbability_TEC6 = cms.double(1.0),
    HitFindingProbability_TEC7 = cms.double(1.0),

    # TIB
    TIB1x = cms.double(0.00195),
    TIB1y = cms.double(3.3775), ## 11.7/sqrt(12.)
    TIB2x = cms.double(0.00191),
    TIB2y = cms.double(3.3775), ## 11.7/sqrt(12.)
    TIB3x = cms.double(0.00325),
    TIB3y = cms.double(3.3775), ## 11.7/sqrt(12.)
    TIB4x = cms.double(0.00323),
    TIB4y = cms.double(3.3775), ## 11.7/sqrt(12.)

    # TID
    TID1x = cms.double(0.00262),
    TID1y = cms.double(3.6662), ## 12.7/sqrt(12.)
    TID2x = cms.double(0.00354),
    TID2y = cms.double(3.6662), ## 12.7/sqrt(12.)
    TID3x = cms.double(0.00391),
    TID3y = cms.double(3.4352), ## 11.9/sqrt(12.)

    # TOB
    TOB1x = cms.double(0.00461),
    TOB1y = cms.double(5.2836), ## 2*9.1514/sqrt(12.)
    TOB2x = cms.double(0.00458),
    TOB2y = cms.double(5.2836), ## 2*9.1514/sqrt(12.)
    TOB3x = cms.double(0.00488),
    TOB3y = cms.double(5.2836), ## 2*9.1514/sqrt(12.)
    TOB4x = cms.double(0.00491),
    TOB4y = cms.double(5.2836), ## 2*9.1514/sqrt(12.)
    TOB5x = cms.double(0.00293),
    TOB5y = cms.double(5.2836), ## 2*9.1514/sqrt(12.)
    TOB6x = cms.double(0.00299),
    TOB6y = cms.double(5.2836), ## 2*9.1514/sqrt(12.)

    # TEC
    TEC1x = cms.double(0.00262),
    TEC1y = cms.double(3.6662), ## 12.7/sqrt(12.)
    TEC2x = cms.double(0.00354),
    TEC2y = cms.double(3.6662), ## 12.7/sqrt(12.)
    TEC3x = cms.double(0.00391),
    TEC3y = cms.double(3.4352), ## 11.9/sqrt(12.)
    TEC4x = cms.double(0.00346),
    TEC4y = cms.double(3.493), ## 12.1/sqrt(12.)
    TEC5x = cms.double(0.00378),
    TEC5y = cms.double(7.1014), ## 2*12.3/sqrt(12.)
    TEC6x = cms.double(0.00508),
    TEC6y = cms.double(6.8704), ## 2*11.9/sqrt(12.)
    TEC7x = cms.double(0.00422),
    TEC7y = cms.double(6.9859), ## 2*12.1/sqrt(12.)


)

