import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.BXlumiParameters_cfi import BXlumiSetup

TrackMon = cms.EDAnalyzer("TrackingMonitor",
    
    # input tags
    numCut           = cms.string(" pt >= 1 & quality('highPurity') "),
    denCut           = cms.string(" pt >= 1 "),
    allTrackProducer = cms.InputTag("generalTracks"),
    TrackProducer    = cms.InputTag("generalTracks"),
    SeedProducer     = cms.InputTag("initialStepSeeds"),
    TCProducer       = cms.InputTag("initialStepTrackCandidates"),
    MVAProducers     = cms.vstring("initialStepClassifier1", "initialStepClassifier2"),
    TrackProducerForMVA = cms.InputTag("initialStepTracks"),
    ClusterLabels    = cms.vstring('Tot'), # to decide which Seeds-Clusters correlation plots to have default is Total other options 'Strip', 'Pix'
    beamSpot         = cms.InputTag("offlineBeamSpot"),
    primaryVertex    = cms.InputTag('offlinePrimaryVertices'),
    stripCluster     = cms.InputTag('siStripClusters'),
    pixelCluster     = cms.InputTag('siPixelClusters'),                          
    BXlumiSetup      = BXlumiSetup.clone(),                              
    genericTriggerEventPSet = cms.PSet(),
#    lumi             = cms.InputTag('lumiProducer'),
#  # taken from 
#  # DPGAnalysis/SiStripTools/src/DigiLumiCorrHistogramMaker.cc
#  # the scale factor 6.37 should follow the lumi prescriptions
#  # AS SOON AS THE CORRECTED LUMI WILL BE AVAILABLE IT HAS TO BE SET TO 1.
#    lumiScale     = cms.double(6.37),
                          
    # PU monitoring
    primaryVertexInputTags    = cms.VInputTag(),
    selPrimaryVertexInputTags = cms.VInputTag(),
    pvLabels = cms.vstring(),
                          
    # output parameters
    AlgoName            = cms.string('GenTk'),
    Quality             = cms.string(''),
    FolderName          = cms.string('Tracking/GlobalParameters'),
    BSFolderName        = cms.string('Tracking/ParametersVsBeamSpot'),
    PVFolderName        = cms.string('Tracking/PrimaryVertices'),

    # determines where to evaluate track parameters
    # options: 'default'      --> straight up track parametes
    #		   'ImpactPoint'  --> evalutate at impact point 
    #		   'InnerSurface' --> evalutate at innermost measurement state 
    #		   'OuterSurface' --> evalutate at outermost measurement state 

    MeasurementState = cms.string('ImpactPoint'),
    
    # which plots to do
    doTestPlots                         = cms.bool(False),
    doAllPlots                          = cms.bool(True),
    doTrackerSpecific                   = cms.bool(False),
    doBeamSpotPlots                     = cms.bool(False),
    doPrimaryVertexPlots                = cms.bool(False),
    doSeedParameterHistos               = cms.bool(False),
    doTrackCandHistos                   = cms.bool(False),
    doAllTrackCandHistos                = cms.bool(False),
    doDCAPlots                          = cms.bool(False),
    doDCAwrtPVPlots                     = cms.bool(False),
    doDCAwrt000Plots                    = cms.bool(False),
    doSIPPlots                          = cms.bool(False),
    doEffFromHitPatternVsPU             = cms.bool(False),
    doEffFromHitPatternVsBX             = cms.bool(False),
    doEffFromHitPatternVsLUMI           = cms.bool(False),
    pvNDOF                              = cms.int32(4),
    pixelCluster4lumi                   = cms.InputTag('siPixelClustersPreSplitting'),
    scal                                = cms.InputTag('scalersRawToDigi'),
    useBPixLayer1                       = cms.bool(False),
    minNumberOfPixelsPerCluster         = cms.int32(2), # from DQM/PixelLumi/python/PixelLumiDQM_cfi.py
    minPixelClusterCharge               = cms.double(15000.),
    doGeneralPropertiesPlots            = cms.bool(False),
    doHitPropertiesPlots                = cms.bool(False),              
#    doGoodTrackPlots                    = cms.bool(False),
    doMeasurementStatePlots             = cms.bool(True),
    doProfilesVsLS                      = cms.bool(False),
    doRecHitsPerTrackProfile            = cms.bool(True),              
    doRecHitVsPhiVsEtaPerTrack          = cms.bool(False),
    doRecHitVsPtVsEtaPerTrack           = cms.bool(False),
#    doGoodTrackRecHitVsPhiVsEtaPerTrack = cms.bool(False),                          
    doLayersVsPhiVsEtaPerTrack          = cms.bool(False),
#    doGoodTrackLayersVsPhiVsEtaPerTrack = cms.bool(False),
#    doGoodTrack2DChi2Plots              = cms.bool(False),
    doThetaPlots                        = cms.bool(False),
    doTrackPxPyPlots                    = cms.bool(False),
    doPUmonitoring                      = cms.bool(False),
    doPlotsVsBXlumi                     = cms.bool(False),
    doPlotsVsGoodPVtx                   = cms.bool(True),
    doPlotsVsLUMI                       = cms.bool(False),
    doPlotsVsBX                         = cms.bool(False),
    doHIPlots                           = cms.bool(False),                              
    doMVAPlots                          = cms.bool(False),
    qualityString = cms.string("highPurity"),                      
    #which seed plots to do
    doSeedNumberHisto = cms.bool(False),
    doSeedLumiAnalysis = cms.bool(False),
    doSeedVsClusterHisto = cms.bool(False),
    doSeedPTHisto = cms.bool(False),
    doSeedETAHisto = cms.bool(False),
    doSeedPHIHisto = cms.bool(False),
    doSeedPHIVsETAHisto = cms.bool(False),
    doSeedThetaHisto = cms.bool(False),
    doSeedQHisto = cms.bool(False),
    doSeedDxyHisto= cms.bool(False),
    doSeedDzHisto= cms.bool(False),
    doSeedNRecHitsHisto= cms.bool(False),
    doSeedNVsPhiProf= cms.bool(False),
    doSeedNVsEtaProf= cms.bool(False),
    doStopSource = cms.bool(False),

    TTRHBuilder = cms.string('WithTrackAngle'),

    # Luminosity based analysis
    doLumiAnalysis = cms.bool(False),                       
    # For plots vs LS
    LSBin = cms.int32(2000),
    LSMin = cms.double(0),
    LSMax = cms.double(2000.),

    # paramters of the Track
    # ============================================================ 
    
    # chi2
    Chi2Bin = cms.int32(50),
    Chi2Max = cms.double(199.5),
    Chi2Min = cms.double(-0.5),

    # chi2 dof
    Chi2NDFBin = cms.int32(50),
    Chi2NDFMax = cms.double(19.5),
    Chi2NDFMin = cms.double(-0.5),

    # chi^2 probability
    Chi2ProbBin = cms.int32(100),
    Chi2ProbMax = cms.double(1.0),
    Chi2ProbMin = cms.double(0.0),

    # Number of Tracks per Event
    TkSizeBin = cms.int32(100),
    TkSizeMax = cms.double(99.5),                        
    TkSizeMin = cms.double(-0.5),

    # Number of seeds per Event
    TkSeedSizeBin = cms.int32(200),
    TkSeedSizeMax = cms.double(999.5),                        
    TkSeedSizeMin = cms.double(-0.5),

    # Number of Track Cadidates per Event
    TCSizeBin = cms.int32(200),
    TCSizeMax = cms.double(999.5),
    TCSizeMin = cms.double(-0.5),

    # num rec hits
    TrackQBin = cms.int32(8),
    TrackQMax = cms.double(2.5),
    TrackQMin = cms.double(-2.5),

    # num rec hits in seed
    SeedHitBin = cms.int32(6),
    SeedHitMax = cms.double(5.5),
    SeedHitMin = cms.double(-0.5),

    # num rec hits per track candidate
    TCHitBin = cms.int32(40),
    TCHitMax = cms.double(39.5),
    TCHitMin = cms.double(-0.5),

    # num rec hits
    RecHitBin = cms.int32(40),
    RecHitMax = cms.double(39.5),                        
    RecHitMin = cms.double(-0.5),

    # mean rec hits
    MeanHitBin = cms.int32(30),
    MeanHitMax = cms.double(29.5),
    MeanHitMin = cms.double(-0.5),

    subdetectors = cms.vstring( "TIB", "TOB", "TID", "TEC", "PixBarrel", "PixEndcap", "Pixel", "Strip" ),
    subdetectorBin = cms.int32(25),

    # num rec hits lost
    RecLostBin = cms.int32(10),
    RecLostMax = cms.double(9.5),
    RecLostMin = cms.double(-0.5),

    # num layers 
    RecLayBin = cms.int32(25),
    RecLayMax = cms.double(24.5),
    RecLayMin = cms.double(-0.5),

    # mean layers
    MeanLayBin = cms.int32(25),
    MeanLayMax = cms.double(24.5),
    MeanLayMin = cms.double(-0.5),

    # num TOB Layers
    TOBLayBin = cms.int32(10),
    TOBLayMax = cms.double(9.5),
    TOBLayMin = cms.double(-0.5),

    # num TIB Layers
    TIBLayBin = cms.int32(6),
    TIBLayMax = cms.double(5.5),
    TIBLayMin = cms.double(-0.5),

    # num TID Layers
    TIDLayBin = cms.int32(6),
    TIDLayMax = cms.double(5.5),
    TIDLayMin = cms.double(-0.5),

    # num TEC Layers
    TECLayBin = cms.int32(15),
    TECLayMax = cms.double(14.5),
    TECLayMin = cms.double(-0.5),

    # num PXB Layers
    PXBLayBin = cms.int32(6),
    PXBLayMax = cms.double(5.5),
    PXBLayMin = cms.double(-0.5),

    # num PXF Layers
    PXFLayBin = cms.int32(6),
    PXFLayMax = cms.double(5.5),
    PXFLayMin = cms.double(-0.5),

    # Track |p|
    TrackPBin = cms.int32(100),
    TrackPMax = cms.double(100),
    TrackPMin = cms.double(0),

    # Track pT
    TrackPtBin = cms.int32(100),
    TrackPtMax = cms.double(100),                        
    TrackPtMin = cms.double(0.1),
    
    # Track px
    TrackPxBin = cms.int32(50),
    TrackPxMax = cms.double(50.0),                        
    TrackPxMin = cms.double(-50.0),
    
    # Track py
    TrackPyBin = cms.int32(50),
    TrackPyMax = cms.double(50.0),                        
    TrackPyMin = cms.double(-50.0),
    
    # Track pz
    TrackPzBin = cms.int32(50),
    TrackPzMax = cms.double(50.0),                        
    TrackPzMin = cms.double(-50.0),
                            
    # track theta
    ThetaBin = cms.int32(32),
    ThetaMax = cms.double(3.2),
    ThetaMin = cms.double(0.0),

    # track eta
    EtaBin = cms.int32(26),
    EtaMax = cms.double(2.5),
    EtaMin = cms.double(-2.5),

    # track phi
    PhiBin = cms.int32(32),
    PhiMax = cms.double(3.141592654),
    PhiMin = cms.double(-3.141592654),

    # Track |p|	error
    pErrBin = cms.int32(50),
    pErrMax = cms.double(1.0),
    pErrMin = cms.double(0.0),

    # Track pT error
    ptErrBin = cms.int32(50),
    ptErrMax = cms.double(1.0),
    ptErrMin = cms.double(0.0),
    
    # Track px error
    pxErrBin = cms.int32(50),
    pxErrMax = cms.double(1.0),
    pxErrMin = cms.double(0.0),
    
    # Track py error
    pyErrBin = cms.int32(50),
    pyErrMax = cms.double(1.0),
    pyErrMin = cms.double(0.0),
    
    # Track pz error
    pzErrBin = cms.int32(50),
    pzErrMax = cms.double(1.0),
    pzErrMin = cms.double(0.0),

    # track eta error
    etaErrBin = cms.int32(50),
    etaErrMax = cms.double(0.1),
    etaErrMin = cms.double(0.0),
    
    # track phi Error
    phiErrBin = cms.int32(50),
    phiErrMax = cms.double(0.1),
    phiErrMin = cms.double(0.0),

    # PCA x position
    VXBin = cms.int32(100),
    VXMax = cms.double(0.5),                        
    VXMin = cms.double(-0.5),
    
    # PCA y position
    VYBin = cms.int32(100),
    VYMax = cms.double(0.5),                        
    VYMin = cms.double(-0.5),
    
    # PCA z position
    VZBin = cms.int32(100),
    VZMax = cms.double(30.0),                        
    VZMin = cms.double(-30.0),
    
    # PCA z position for profile
    VZBinProf = cms.int32(100),
    VZMaxProf = cms.double(0.2),                        
    VZMinProf = cms.double(-0.2),
    
    # PCA x position for 2D plot
    X0Bin = cms.int32(100),
    X0Max = cms.double(0.5),                        
    X0Min = cms.double(-0.5),
    
    # PCA y position for 2D plot
    Y0Bin = cms.int32(100),
    Y0Max = cms.double(0.5),                        
    Y0Min = cms.double(-0.5),
    
    # PCA z position for 2D plot
    Z0Bin = cms.int32(120),
    Z0Max = cms.double(60.0),                        
    Z0Min = cms.double(-60.0),
    
    # Track dxy (transverse impact parameter)
    DxyBin = cms.int32(100),
    DxyMax = cms.double(0.5),
    DxyMin = cms.double(-0.5),                        

    AbsDxyBin = cms.int32(120),
    AbsDxyMin = cms.double(0.),
    AbsDxyMax = cms.double(60.),                        

    # Seed dxy (transverse impact parameter)
    SeedDxyBin = cms.int32(100),
    SeedDxyMax = cms.double(0.5),
    SeedDxyMin = cms.double(-0.5),                        

    # Seed dz (longitudinal impact parameter)
    SeedDzBin = cms.int32(120),
    SeedDzMax = cms.double(30.0),
    SeedDzMin = cms.double(-30.0),                        

    # Track Candidate dxy (transverse impact parameter)
    TCDxyBin = cms.int32(100),
    TCDxyMax = cms.double(100.0),
    TCDxyMin = cms.double(-100.0),                                                

    # Track Candidate dz (transverse impact parameter)
    TCDzBin = cms.int32(100),
    TCDzMax = cms.double(400.0),
    TCDzMin = cms.double(-400.0),                                                

    # Track selection MVA
    MVABin  = cms.int32(100),
    MVAMin  = cms.double(-1),
    MVAMax  = cms.double(1),

#######################################
## needed for tracksVScluster and seedVScluster

    # NCluster Pixel
    NClusPxBin = cms.int32(200),
    NClusPxMax = cms.double(19999.5),                      
    NClusPxMin = cms.double(-0.5),

    # NCluster Strip
    NClusStrBin = cms.int32(500),
    NClusStrMax = cms.double(99999.5),                      
    NClusStrMin = cms.double(-0.5),

    # NCluster Vs Tracks
    NTrk2DBin     = cms.int32(50),
    NTrk2DMax     = cms.double(1999.5),                      
    NTrk2DMin     = cms.double(-0.5),

    # PU monitoring
    # Nunmber of Good Primary Vertices
    GoodPVtxBin = cms.int32(200),
    GoodPVtxMin = cms.double( 0.),
    GoodPVtxMax = cms.double(200.),

    LUMIBin  = cms.int32 ( 300 ),   # irrelevant
    LUMIMin  = cms.double(  200.),
    LUMIMax  = cms.double(20000.),

#    # BXlumi                          
#    BXlumiBin = cms.int32(400),
#    BXlumiMin = cms.double(4000),
#    BXlumiMax = cms.double(20000),

###############################
################## FOR HI PLOTS#####################
#######
TransDCABins = cms.int32(100),
TransDCAMin = cms.double(-8.0),
TransDCAMax = cms.double(8.0),

LongDCABins = cms.int32(100),
LongDCAMin = cms.double(-8.0),
LongDCAMax = cms.double(8.0),          
)

# Overcoming the 255 arguments limit
# TrackingRegion monitoring
TrackMon.RegionProducer = cms.InputTag("")
TrackMon.RegionCandidates = cms.InputTag("")
TrackMon.doRegionPlots = cms.bool(False)
TrackMon.RegionSizeBin = cms.int32(20)
TrackMon.RegionSizeMax = cms.double(19.5)
TrackMon.RegionSizeMin = cms.double(-0.5)
TrackMon.RegionCandidatePtBin = cms.int32(100)
TrackMon.RegionCandidatePtMax = cms.double(1000)
TrackMon.RegionCandidatePtMin = cms.double(0)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase1Pixel.toModify(TrackMon, EtaBin=30, EtaMin=-3, EtaMax=3)
phase2_tracker.toModify(TrackMon, EtaBin=46, EtaMin=-4.5, EtaMax=4.5)
