import FWCore.ParameterSet.Config as cms

siTrackerGaussianSmearingRecHits = cms.EDProducer("SiTrackerGaussianSmearingRecHitConverter",

    #converting energy loss from GeV to ADC counts
    GevPerElectron = cms.double(3.61e-09),
    ElectronsPerADC = cms.double(250.0),
             
    HitFindingProbability_TEC3 = cms.double(1.0),
    HitFindingProbability_TEC4 = cms.double(1.0),
    TIB1y = cms.double(3.3775), ## 11.7/sqrt(12.)

    # TIB
    TIB1x = cms.double(0.00195),
    # matching of 1dim hits in double-sided modules
    # creating 2dim hits
    doRecHitMatching = cms.bool(True),

    # Set to (True) for taking the existence of dead modules into account:
    killDeadChannels = cms.bool(True),
                                                  
    TEC2y = cms.double(3.6662), ## 12.7/sqrt(12.)

    TEC2x = cms.double(0.00354),
    #
    DeltaRaysMomentumCut = cms.double(0.5),
    AlphaForward_BinNNew = cms.int32(0),
    PixelBarrelResolutionFile40T = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution40T.root'),
    PixelBarrelResolutionFile38T = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution38T.root'),
    BetaForward_BinMin = cms.double(0.0),
    HitFindingProbability_TID1 = cms.double(1.0),
    HitFindingProbability_TID3 = cms.double(1.0),
    HitFindingProbability_TID2 = cms.double(1.0),
    AlphaBarrel_BinMin = cms.double(-0.2),
    PixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution.root'),
    TOB4x = cms.double(0.00491),
    TIB2y = cms.double(3.3775), ## 11.7/sqrt(12.)

    TID2x = cms.double(0.00354),
    TID2y = cms.double(3.6662), ## 12.7/sqrt(12.)

    # TEC
    TEC1x = cms.double(0.00262),
    BetaForwardMultiplicity = cms.int32(3),
    AlphaForward_BinWidthNew = cms.double(0.0),
    AlphaForward_BinWidth = cms.double(0.0),
    # If you want to have RecHits == PSimHits (tracking with PSimHits)
    trackingPSimHits = cms.bool(False),
    BetaForward_BinWidthNew = cms.double(0.0),
    AlphaBarrel_BinWidthNew = cms.double(0.1),
    AlphaBarrel_BinN = cms.int32(4),
    HitFindingProbability_TOB5 = cms.double(1.0),
    AlphaBarrelMultiplicityNew = cms.int32(4),
    UseNewParametrization = cms.bool(True),
    TEC4y = cms.double(3.493), ## 12.1/sqrt(12.)

    TEC4x = cms.double(0.00346),
    AlphaBarrel_BinWidth = cms.double(0.1),
    TOB1y = cms.double(5.2836), ## 2*9.1514/sqrt(12.)

    # TOB
    TOB1x = cms.double(0.00461),
    AlphaForward_BinN = cms.int32(0),
    AlphaForward_BinMinNew = cms.double(0.0),
    # Pixel
    PixelMultiplicityFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelData.root'),
    HitFindingProbability_TOB2 = cms.double(1.0),
    AlphaForward_BinMin = cms.double(0.0),
    TOB6x = cms.double(0.00299),
    TOB6y = cms.double(5.2836), ## 2*9.1514/sqrt(12.)

    TEC3x = cms.double(0.00391),
    TEC3y = cms.double(3.4352), ## 11.9/sqrt(12.)

    BetaForward_BinWidth = cms.double(0.0),
    BetaBarrelMultiplicity = cms.int32(6),
    AlphaBarrelMultiplicity = cms.int32(4),
    TIB4x = cms.double(0.00323),
    TOB2y = cms.double(5.2836), ## 2*9.1514/sqrt(12.)

    BetaForward_BinN = cms.int32(0),
    TEC6y = cms.double(6.8704), ## 2*11.9/sqrt(12.)

    TEC6x = cms.double(0.00508),
    TOB3y = cms.double(5.2836), ## 2*9.1514/sqrt(12.)

    TOB3x = cms.double(0.00488),
    TID1y = cms.double(3.6662), ## 12.7/sqrt(12.)

    # TID
    TID1x = cms.double(0.00262),
    BetaBarrel_BinWidthNew = cms.double(0.2),
    BetaForward_BinNNew = cms.int32(0),
    BetaBarrel_BinN = cms.int32(7),
    AlphaForwardMultiplicity = cms.int32(3),
    # CMSSW
    #vstring ROUList = { "TrackerHitsPixelBarrelLowTof","TrackerHitsPixelBarrelHighTof",
    #	"TrackerHitsPixelEndcapLowTof","TrackerHitsPixelEndcapHighTof",
    #	"TrackerHitsTIBLowTof","TrackerHitsTIBHighTof","TrackerHitsTIDLowTof","TrackerHitsTIDHighTof",
    #	"TrackerHitsTOBLowTof","TrackerHitsTOBHighTof","TrackerHitsTECLowTof","TrackerHitsTECHighTof",
    #   "TrackerHitsPixelBarrelLowTof","TrackerHitsPixelBarrelHighTof",
    #   "TrackerHitsPixelEndcapLowTof","TrackerHitsPixelEndcapHighTof" } 	
    # FAMOS
    ROUList = cms.VInputTag(cms.InputTag("mix","famosSimHitsTrackerHits")),
    PixelBarrelResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelBarrelResolution.root'),
    TOB5y = cms.double(5.2836), ## 2*9.1514/sqrt(12.)

    TEC5x = cms.double(0.00378),
    TEC5y = cms.double(7.1014), ## 2*12.3/sqrt(12.)

    BetaForwardMultiplicityNew = cms.int32(3),
    TOB5x = cms.double(0.00293),
    BetaBarrel_BinNNew = cms.int32(7),
    AlphaBarrel_BinNNew = cms.int32(4),
    AlphaForwardMultiplicityNew = cms.int32(3),
    TIB3y = cms.double(3.3775), ## 11.7/sqrt(12.)

    TIB3x = cms.double(0.00325),
    UseSigma = cms.bool(True),
    BetaForward_BinMinNew = cms.double(0.0),
    TID3y = cms.double(3.4352), ## 11.9/sqrt(12.)

    TID3x = cms.double(0.00391),
    BetaBarrelMultiplicityNew = cms.int32(7),
    BetaBarrel_BinMin = cms.double(0.0),
    HitFindingProbability_TEC1 = cms.double(1.0),
    HitFindingProbability_TEC2 = cms.double(1.0),
    BetaBarrel_BinWidth = cms.double(0.2),
    # Switch between old and new parametrization
    UseCMSSWPixelParametrization = cms.bool(True),
    HitFindingProbability_TEC5 = cms.double(1.0),
    HitFindingProbability_TEC6 = cms.double(1.0),
    HitFindingProbability_TEC7 = cms.double(1.0),
    TEC1y = cms.double(3.6662), ## 12.7/sqrt(12.)

    VerboseLevel = cms.untracked.int32(2),
    HitFindingProbability_TIB3 = cms.double(1.0),
    HitFindingProbability_TIB2 = cms.double(1.0),
    HitFindingProbability_TIB1 = cms.double(1.0),
    HitFindingProbability_TOB6 = cms.double(1.0),
    HitFindingProbability_TOB1 = cms.double(1.0),
    HitFindingProbability_TOB3 = cms.double(1.0),
    HitFindingProbability_TIB4 = cms.double(1.0),
    # Pixel CMSSW Parametrization
    PixelMultiplicityFile40T = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelData40T.root'),
    PixelMultiplicityFile38T = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelData38T.root'),

    TEC7x = cms.double(0.00422),
    TEC7y = cms.double(6.9859), ## 2*12.1/sqrt(12.)

    PixelForwardResolutionFile40T = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution40T.root'),
    PixelForwardResolutionFile38T = cms.string('FastSimulation/TrackingRecHitProducer/data/PixelForwardResolution38T.root'),
    TOB2x = cms.double(0.00458),
    TIB4y = cms.double(3.3775), ## 11.7/sqrt(12.)

    # Needed to compute Pixel Errors
    PixelErrorParametrization = cms.string('NOTcmsim'),
    BetaBarrel_BinMinNew = cms.double(0.0),
    # Hit Finding Probabilities
    HitFindingProbability_PXB = cms.double(1.0),
    AlphaBarrel_BinMinNew = cms.double(-0.2),
    TIB2x = cms.double(0.00191),
    HitFindingProbability_TOB4 = cms.double(1.0),
    TOB4y = cms.double(5.2836), ## 2*9.1514/sqrt(12.)

    HitFindingProbability_PXF = cms.double(1.0),

    templateIdBarrel = cms.int32( 20 ),
    templateIdForward  = cms.int32( 21 ),
    NewPixelForwardResolutionFile = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionForward38T.root'),
    NewPixelBarrelResolutionFile1 = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionBarrel38T.root'),
    NewPixelBarrelResolutionFile2 = cms.string('FastSimulation/TrackingRecHitProducer/data/NewPixelResolutionBarrelEdge38T.root')


)

