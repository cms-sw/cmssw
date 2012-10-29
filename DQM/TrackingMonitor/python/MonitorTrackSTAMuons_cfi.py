import FWCore.ParameterSet.Config as cms

# MonitorTrackGlobal
MonitorTrackSTAMuons = cms.EDAnalyzer("TrackingMonitor",
    # input tags
    TrackProducer = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
    SeedProducer  = cms.InputTag("combinedP5SeedsForCTF"),
    TCProducer    = cms.InputTag("ckfTrackCandidatesP5"),
    beamSpot      = cms.InputTag("offlineBeamSpot"),
    ClusterLabels = cms.vstring('Tot'),
                                      
    # output parameters
    OutputMEsInRootFile = cms.bool(False),
    AlgoName            = cms.string('sta'),
    Quality             = cms.string(''),
    OutputFileName      = cms.string('monitortrackparameters_stamuons.root'),
    FolderName          = cms.string('Muons/standAloneMuonsUpdatedAtVtx'),
    BSFolderName        = cms.string('Muons/standAloneMuonsUpdatedAtVtx/BeamSpotParameters'),

    # determines where to evaluate track parameters
    # options: 'default'      --> straight up track parametes
    #		   'ImpactPoint'  --> evalutate at impact point 
    #		   'InnerSurface' --> evalutate at innermost measurement state 
    #		   'OuterSurface' --> evalutate at outermost measurement state 
    MeasurementState = cms.string('default'),

    # which plots to do
    doTrackerSpecific          = cms.bool(False),
    doAllPlots                 = cms.bool(False),
    doBeamSpotPlots            = cms.bool(False),
    doSeedParameterHistos      = cms.bool(False),
    doTrackCandHistos          = cms.bool(False),
    doDCAPlots                 = cms.bool(False),
    doGeneralPropertiesPlots   = cms.bool(True),
    doHitPropertiesPlots       = cms.bool(True),              
    doGoodTrackPlots           = cms.bool(False),
    doMeasurementStatePlots    = cms.bool(True),
    doProfilesVsLS             = cms.bool(False),
    doRecHitVsPhiVsEtaPerTrack = cms.bool(False),
    doGoodTrackRecHitVsPhiVsEtaPerTrack = cms.bool(False),                          

    #which seed plots to do
    doSeedNumberHisto = cms.bool(False),
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


    # paramters of the Track
    # ============================================================ 
    
    # chi2
    Chi2Bin = cms.int32(250),
    Chi2Max = cms.double(500.0),
    Chi2Min = cms.double(0.0),

    # chi2 dof
    Chi2NDFBin = cms.int32(200),
    Chi2NDFMax = cms.double(19.5),
    Chi2NDFMin = cms.double(-0.5),

    # chi^2 probability
    Chi2ProbBin = cms.int32(100),
    Chi2ProbMax = cms.double(1.0),
    Chi2ProbMin = cms.double(0.0),

    # Number of Tracks per Event
    TkSizeBin = cms.int32(11),
    TkSizeMax = cms.double(10.5),
    TkSizeMin = cms.double(-0.5),

    # Number of seeds per Event                                    
    TkSeedSizeBin = cms.int32(20),
    TkSeedSizeMax = cms.double(19.5),                                    
    TkSeedSizeMin = cms.double(-0.5),

    # Number of Track Cadidates per Event
    TCSizeBin = cms.int32(150),
    TCSizeMax = cms.double(149.5),                                    
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
    RecHitBin = cms.int32(120),
    RecHitMax = cms.double(120.0),
    RecHitMin = cms.double(0.0),

    # mean rec hits
    MeanHitBin = cms.int32(30),
    MeanHitMax = cms.double(29.5),                                    
    MeanHitMin = cms.double(-0.5),

    # num TOB rec hits
    TOBHitBin = cms.int32(15),
    TOBHitMin = cms.double(-0.5),
    TOBHitMax = cms.double(14.5),

    # num TIB rec hits
    TIBHitBin = cms.int32(15),
    TIBHitMin = cms.double(-0.5),
    TIBHitMax = cms.double(14.5),

    # num TID rec hits
    TIDHitBin = cms.int32(15),
    TIDHitMin = cms.double(-0.5),
    TIDHitMax = cms.double(14.5),

    # num TEC rec hits
    TECHitBin = cms.int32(25),
    TECHitMin = cms.double(-0.5),
    TECHitMax = cms.double(24.5),

    # num PXB rec hits
    PXBHitBin = cms.int32(10),
    PXBHitMin = cms.double(-0.5),
    PXBHitMax = cms.double(9.5),

    # num PXF rec hits
    PXFHitBin = cms.int32(10),
    PXFHitMin = cms.double(-0.5),
    PXFHitMax = cms.double(9.5),

    # num rec hits lost
    RecLostBin = cms.int32(120),
    RecLostMax = cms.double(20),                                    
    RecLostMin = cms.double(0.0),

    # num layers
    RecLayBin = cms.int32(120),
    RecLayMax = cms.double(120.0),
    RecLayMin = cms.double(0.0),

    # mean layers 
    MeanLayBin = cms.int32(20),
    MeanLayMax = cms.double(19.5),                                    
    MeanLayMin = cms.double(-0.5),

    # num TOB layers
    TOBLayBin = cms.int32(10),
    TOBLayMax = cms.double(9.5),                                    
    TOBLayMin = cms.double(-0.5),

    # num TIB layers
    TIBLayBin = cms.int32(6),
    TIBLayMax = cms.double(5.5),                                    
    TIBLayMin = cms.double(-0.5),

    # num TID layers
    TIDLayBin = cms.int32(6),
    TIDLayMax = cms.double(5.5),                                    
    TIDLayMin = cms.double(-0.5),

    # num TEC layers
    TECLayBin = cms.int32(15),
    TECLayMax = cms.double(14.5),                                    
    TECLayMin = cms.double(-0.5),

    # num PXB layers
    PXBLayBin = cms.int32(6),
    PXBLayMax = cms.double(5.5),                                    
    PXBLayMin = cms.double(-0.5),

    # num PXF layers
    PXFLayBin = cms.int32(6),
    PXFLayMax = cms.double(5.5),                                    
    PXFLayMin = cms.double(-0.5),

    # Track |p|	
    TrackPBin = cms.int32(1000),
    TrackPMax = cms.double(1000),                                    
    TrackPMin = cms.double(0),

    # Track pT
    TrackPtBin = cms.int32(1000),
    TrackPtMax = cms.double(1000),
    TrackPtMin = cms.double(0),

    # Track px 
    TrackPxBin = cms.int32(1000),
    TrackPxMax = cms.double(500.0),
    TrackPxMin = cms.double(-500.0),

    # Track py
    TrackPyBin = cms.int32(1000),
    TrackPyMax = cms.double(500.0),
    TrackPyMin = cms.double(-500.0),

    # Track pz
    TrackPzMin = cms.double(-500.0),
    TrackPzMax = cms.double(500.0),
    TrackPzBin = cms.int32(1000),

    # track theta
    ThetaBin = cms.int32(100),
    ThetaMax = cms.double(3.2),
    ThetaMin = cms.double(0.0),
                                    
    # track eta
    EtaBin = cms.int32(100),
    EtaMax = cms.double(3.0),
    EtaMin = cms.double(-3.0),
                                    
    # track phi
    PhiBin = cms.int32(36),
    PhiMax = cms.double(3.2),
    PhiMin = cms.double(-3.2),
                                    
    # Track |p|	error
    pErrBin = cms.int32(100),
    pErrMax = cms.double(10.0),
    pErrMin = cms.double(0.0),
                                    
    # Track pT error
    ptErrBin = cms.int32(100),
    ptErrMax = cms.double(10.0),
    ptErrMin = cms.double(0.0),
                                    
    # Track px error
    pxErrBin = cms.int32(100),
    pxErrMax = cms.double(10.0),
    pxErrMin = cms.double(0.0),
                                    
    # Track py error
    pyErrBin = cms.int32(100),
    pyErrMax = cms.double(10.0),
    pyErrMin = cms.double(0.0),
                                    
    # Track pz error
    pzErrBin = cms.int32(100),
    pzErrMax = cms.double(10.0),
    pzErrMin = cms.double(0.0),

    # track eta error
    etaErrBin = cms.int32(100),
    etaErrMax = cms.double(0.5),
    etaErrMin = cms.double(0.0),
                                    
    # track phi Error
    phiErrBin = cms.int32(100),                                    
    phiErrMax = cms.double(1.0),
    phiErrMin = cms.double(0.0),

    # PCA x position
    VXBin = cms.int32(20),
    VXMax = cms.double(20.0),
    VXMin = cms.double(-20.0),

    # PCA y position   
    VYBin = cms.int32(20),
    VYMax = cms.double(20.0),
    VYMin = cms.double(-20.0),

    # PCA z position
    VZBin = cms.int32(50),
    VZMax = cms.double(100.0),
    VZMin = cms.double(-100.0),
                                    
    # PCA x position for 2D plot
    X0Bin = cms.int32(100),
    X0Max = cms.double(3.0),                                    
    X0Min = cms.double(-3.0),

    # PCA y position for 2D plot
    Y0Bin = cms.int32(100),
    Y0Max = cms.double(3.0),                                    
    Y0Min = cms.double(-3.0),

    # PCA z position for 2D plot
    Z0Bin = cms.int32(60),
    Z0Max = cms.double(30.0),
    Z0Min = cms.double(-30.0),
                                    
    # Track dxy (transverse impact parameter)
    DxyBin = cms.int32(100),
    DxyMax = cms.double(0.5),                                    
    DxyMin = cms.double(-0.5),                        

    # Seed dxy (transverse impact parameter)
    SeedDxyBin = cms.int32(100),
    SeedDxyMax = cms.double(0.5),
    SeedDxyMin = cms.double(-0.5),                                    

    # Seed dz (longitudinal impact parameter)
    SeedDzBin = cms.int32(200),
    SeedDzMax = cms.double(30.0),
    SeedDzMin = cms.double(-30.0),                                                            

    # Track Candidate dxy (transverse impact parameter)
    TCDxyBin = cms.int32(200),
    TCDxyMax = cms.double(100.0),
    TCDxyMin = cms.double(-100.0),                                                

    # Track Candidate dz (transverse impact parameter)
    TCDzBin = cms.int32(200),
    TCDzMax = cms.double(400.0),
    TCDzMin = cms.double(-400.0),                                                

    # NCluster Pixel
    NClusPxBin = cms.int32(50),
    NClusPxMax = cms.double(1999.5),                      
    NClusPxMin = cms.double(-0.5),

    # NCluster Strip
    NClusStrBin = cms.int32(150),
    NClusStrMax = cms.double(14999.5),                      
    NClusStrMin = cms.double(-0.5),

    # NCluster 2D
    NClus2DPxBin  = cms.int32(20),
    NClus2DPxMax  = cms.double(1999.5),                      
    NClus2DPxMin  = cms.double(-0.5),
    NClus2DStrBin = cms.int32(50),
    NClus2DStrMax = cms.double(14999.5),                      
    NClus2DStrMin = cms.double(-0.5),

    # NCluster Vs Tracks
    NClus2DTotBin = cms.int32(50),
    NClus2DTotMax = cms.double(14999.5),                      
    NClus2DTotMin = cms.double(-0.5),
    NTrk2DBin     = cms.int32(20),
    NTrk2DMax     = cms.double(199.5),                      
    NTrk2DMin     = cms.double(-0.5),
                          
    TTRHBuilder = cms.string('WithTrackAngle'),

    # For plots vs LS
    LSBin = cms.int32(2000),
    LSMin = cms.double(0),
    LSMax = cms.double(2000.),
                                
    # Luminosity based analysis
    doLumiAnalysis = cms.bool(False)                       
)
