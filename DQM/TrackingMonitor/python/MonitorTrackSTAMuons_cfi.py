import FWCore.ParameterSet.Config as cms

# MonitorTrackGlobal
from DQM.TrackingMonitor.TrackingMonitor_cfi import *
MonitorTrackSTAMuons = TrackMon.clone(
    # input tags
    TrackProducer = ("standAloneMuons","UpdatedAtVtx"),
    SeedProducer = "combinedP5SeedsForCTF",
    TCProducer = "ckfTrackCandidatesP5",
    beamSpot = "offlineBeamSpot",
    ClusterLabels = ('Tot',),

    # output parameters
    AlgoName = 'sta',
    Quality = '',
    FolderName = 'Muons/standAloneMuonsUpdatedAtVtx',
    BSFolderName = 'Muons/standAloneMuonsUpdatedAtVtx/BeamSpotParameters',

    # determines where to evaluate track parameters
    # options: 'default'      --> straight up track parametes
    #		   'ImpactPoint'  --> evalutate at impact point
    #		   'InnerSurface' --> evalutate at innermost measurement state
    #		   'OuterSurface' --> evalutate at outermost measurement state
    MeasurementState = 'default',

    # which plots to do
    doTrackerSpecific = False,
    doAllPlots = False,
    doBeamSpotPlots = False,
    doSeedParameterHistos = False,
    doTrackCandHistos = False,
    doDCAPlots = False,
    doGeneralPropertiesPlots = True,
    doHitPropertiesPlots = True,
    doEffFromHitPatternVsPU = False,
    doEffFromHitPatternVsBX = False,
    #doGoodTrackPlots = False,
    doMeasurementStatePlots = True,
    doProfilesVsLS = False,
    doRecHitVsPhiVsEtaPerTrack = False,
    #doGoodTrackRecHitVsPhiVsEtaPerTrack = False,

    # which seed plots to do
    doSeedNumberHisto= False,
    doSeedVsClusterHisto = False,
    doSeedPTHisto = False,
    doSeedETAHisto = False,
    doSeedPHIHisto = False,
    doSeedPHIVsETAHisto = False,
    doSeedThetaHisto = False,
    doSeedQHisto = False,
    doSeedDxyHisto = False,
    doSeedDzHisto = False,
    doSeedNRecHitsHisto = False,
    doSeedNVsPhiProf = False,
    doSeedNVsEtaProf = False,

    # paramters of the Track
    # ============================================================

    # chi2
    Chi2Bin = 250,
    Chi2Max = 500.0,
    Chi2Min = 0.0,
    # chi2 dof
    Chi2NDFBin = 200,
    Chi2NDFMax = 19.5,
    Chi2NDFMin = -0.5,
    # chi^2 probability
    Chi2ProbBin = 100,
    Chi2ProbMax = 1.0,
    Chi2ProbMin = 0.0,
    # Number of Tracks per Event
    TkSizeBin = 11,
    TkSizeMax = 10.5,
    TkSizeMin = -0.5,
    # Number of seeds per Event
    TkSeedSizeBin = 20,
    TkSeedSizeMax = 19.5,
    TkSeedSizeMin = -0.5,
    # Number of Track Cadidates per Event
    TCSizeBin = 150,
    TCSizeMax = 149.5,
    TCSizeMin = -0.5,
    # num rec hits
    TrackQBin = 8,
    TrackQMax = 2.5,
    TrackQMin = -2.5,
    # num rec hits in seed
    SeedHitBin = 6,
    SeedHitMax = 5.5,
    SeedHitMin = -0.5,
    # num rec hits per track candidate
    TCHitBin = 40,
    TCHitMax = 39.5,
    TCHitMin = -0.5,
    # num rec hits
    RecHitBin = 120,
    RecHitMax = 120.0,
    RecHitMin = 0.0,
    # mean rec hits
    MeanHitBin = 30,
    MeanHitMax = 29.5,
    MeanHitMin = -0.5,
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
    RecLostBin = 120,
    RecLostMax = 20.,
    RecLostMin = 0.0,
    # num layers
    RecLayBin = 120,
    RecLayMax = 120.0,
    RecLayMin = 0.0,
    # mean layers
    MeanLayBin = 20,
    MeanLayMax = 19.5,
    MeanLayMin = -0.5,
    # num TOB layers
    TOBLayBin = 10,
    TOBLayMax = 9.5,
    TOBLayMin = -0.5,
    # num TIB layers
    TIBLayBin = 6,
    TIBLayMax = 5.5,
    TIBLayMin = -0.5,
    # num TID layers
    TIDLayBin = 6,
    TIDLayMax = 5.5,
    TIDLayMin = -0.5,
    # num TEC layers
    TECLayBin = 15,
    TECLayMax = 14.5,
    TECLayMin = -0.5,
    # num PXB layers
    PXBLayBin = 6,
    PXBLayMax = 5.5,
    PXBLayMin = -0.5,
    # num PXF layers
    PXFLayBin = 6,
    PXFLayMax = 5.5,
    PXFLayMin = -0.5,
    # Track |p|
    TrackPBin = 1000,
    TrackPMax = 1000.,
    TrackPMin = 0.,
    # Track pT
    TrackPtBin = 1000,
    TrackPtMax = 1000.,
    TrackPtMin = 0.,
    # Track px
    TrackPxBin = 1000,
    TrackPxMax = 500.0,
    TrackPxMin = -500.0,
    # Track py
    TrackPyBin = 1000,
    TrackPyMax = 500.0,
    TrackPyMin = -500.0,
    # Track pz
    TrackPzMin = -500.0,
    TrackPzMax = 500.0,
    TrackPzBin = 1000,
    # track theta
    ThetaBin = 100,
    ThetaMax = 3.2,
    ThetaMin = 0.0,
    # track eta
    EtaBin = 100,
    EtaMax = 3.0,
    EtaMin = -3.0,
    # track phi
    PhiBin = 36,
    PhiMax = 3.2,
    PhiMin = -3.2,
    # Track |p|	error
    pErrBin = 100,
    pErrMax = 10.0,
    pErrMin = 0.0,
    # Track pT error
    ptErrBin = 100,
    ptErrMax = 10.0,
    ptErrMin = 0.0,
    # Track px error
    pxErrBin = 100,
    pxErrMax = 10.0,
    pxErrMin = 0.0,
    # Track py error
    pyErrBin = 100,
    pyErrMax = 10.0,
    pyErrMin = 0.0,
    # Track pz error
    pzErrBin = 100,
    pzErrMax = 10.0,
    pzErrMin = 0.0,
    # track eta error
    etaErrBin = 100,
    etaErrMax = 0.5,
    etaErrMin = 0.0,
    # track phi Error
    phiErrBin = 100,
    phiErrMax = 1.0,
    phiErrMin = 0.0,
    # PCA x position
    VXBin = 20,
    VXMax = 20.0,
    VXMin = -20.0,
    # PCA y position
    VYBin = 20,
    VYMax = 20.0,
    VYMin = -20.0,
    # PCA z position
    VZBin = 50,
    VZMax = 100.0,
    VZMin = -100.0,
    # PCA x position for 2D plot
    X0Bin = 100,
    X0Max = 3.0,
    X0Min = -3.0,
    # PCA y position for 2D plot
    Y0Bin = 100,
    Y0Max = 3.0,
    Y0Min = -3.0,
    # PCA z position for 2D plot
    Z0Bin = 60,
    Z0Max = 30.0,
    Z0Min = -30.0,
    # Track dxy (transverse impact parameter)
    DxyBin = 100,
    DxyMax = 0.5,
    DxyMin = -0.5,
    # Seed dxy (transverse impact parameter)
    SeedDxyBin = 100,
    SeedDxyMax = 0.5,
    SeedDxyMin = -0.5,
    # Seed dz (longitudinal impact parameter)
    SeedDzBin = 200,
    SeedDzMax = 30.0,
    SeedDzMin = -30.0,
    # Track Candidate dxy (transverse impact parameter)
    TCDxyBin = 200,
    TCDxyMax = 100.0,
    TCDxyMin = -100.0,
    # Track Candidate dz (transverse impact parameter)
    TCDzBin = 200,
    TCDzMax = 400.0,
    TCDzMin = -400.0,
    # NCluster Pixel
    NClusPxBin = 50,
    NClusPxMax = 1999.5,
    NClusPxMin = -0.5,
    # NCluster Strip
    NClusStrBin = 150,
    NClusStrMax = 14999.5,
    NClusStrMin = -0.5,
    # NCluster 2D
    NClus2DPxBin = cms.int32(20),
    NClus2DPxMax = cms.double(1999.5),
    NClus2DPxMin = cms.double(-0.5),
    NClus2DStrBin = cms.int32(50),
    NClus2DStrMax = cms.double(14999.5),
    NClus2DStrMin = cms.double(-0.5),
    # NCluster Vs Tracks
    NClus2DTotBin = cms.int32(50),
    NClus2DTotMax = cms.double(14999.5),
    NClus2DTotMin = cms.double(-0.5),
    NTrk2D = TrackMon.NTrk2D.clone(
        NTrk2DBin = 20,
        NTrk2DMax = 199.5,
        NTrk2DMin = -0.5
    ),
    TTRHBuilder = 'WithTrackAngle',
    # For plots vs LS
    LSBin = 2000,
    LSMin = 0.,
    LSMax = 2000.,
    # Luminosity based analysis
    doLumiAnalysis = False
)
