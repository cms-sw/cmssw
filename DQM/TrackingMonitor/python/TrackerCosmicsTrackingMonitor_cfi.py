
import FWCore.ParameterSet.Config as cms

TrackerCosmicTrackMon = cms.EDFilter("TrackingMonitor",
    TrackProducer = cms.InputTag("generalTracks"),
    SeedProducer = cms.InputTag("combinedP5SeedsForCTF"),
    TCProducer = cms.InputTag("ckfTrackCandidatesP5"),
    AlgoName = cms.string('GenTk'),
    beamSpot = cms.InputTag("offlineBeamSpot"),                                     
    OutputMEsInRootFile = cms.bool(False),
    OutputFileName = cms.string('MonitorTrack.root'),

    FolderName = cms.string('Track/GlobalParameters'),

    MeasurementState = cms.string('default'),

    doTrackerSpecific = cms.bool(True),
    doAllPlots = cms.bool(False),                    
    doSeedParameterHistos = cms.bool(False),

    Chi2Bin = cms.int32(100),
    Chi2Max = cms.double(500.0),
    Chi2Min = cms.double(-0.5),
 
    Chi2ProbMax = cms.double(1.0),
    Chi2ProbBin = cms.int32(100),
    Chi2ProbMin = cms.double(0.0),


    D0Bin = cms.int32(200),
    D0Max = cms.double(100.0),
    D0Min = cms.double(-100.0),

    EtaBin = cms.int32(100),
    EtaMax = cms.double(3.0),
    EtaMin = cms.double(-3.0),

    etaErrBin = cms.int32(100),
    etaErrMax = cms.double(0.5),
    etaErrMin = cms.double(0.0),

    pErrBin = cms.int32(100),
    pErrMax = cms.double(1000.0),
    pErrMin = cms.double(-1000.0),

    phiErrBin = cms.int32(100),
    phiErrMax = cms.double(0.5),
    phiErrMin = cms.double(0.0),

    PhiBin = cms.int32(36),
    PhiMax = cms.double(3.2),
    PhiMin = cms.double(-3.2),
    
    ptErrBin = cms.int32(100),
    ptErrMax = cms.double(1000.0),
    ptErrMin = cms.double(0.0),
    
    pxErrBin = cms.int32(100),
    pxErrMax = cms.double(1000.0),
    pxErrMin = cms.double(-1000.0),
    
    pyErrBin = cms.int32(100),
    pyErrMax = cms.double(1000.0),
    pyErrMin = cms.double(-1000.0),
    
    pzErrBin = cms.int32(100),
    pzErrMax = cms.double(1000.0),
    pzErrMin = cms.double(-1000.0),
    
    RecHitBin = cms.int32(35),
    RecHitMax = cms.double(34.5),
    RecHitMin = cms.double(-0.5),

    RecLostBin = cms.int32(10),
    RecLostMax = cms.double(9.5),
    RecLostMin = cms.double(-0.5),

    RecLayBin = cms.int32(35),
    RecLayMax = cms.double(34.5),
    RecLayMin = cms.double(-0.5),
    
    ThetaBin = cms.int32(100),
    ThetaMax = cms.double(3.2),
    ThetaMin = cms.double(0.0),
    
    TkSizeBin = cms.int32(25),
    TkSizeMax = cms.double(24.5),
    TkSizeMin = cms.double(-0.5),

    TkSeedSizeBin = cms.int32(20),
    TkSeedSizeMin = cms.double(-0.5),
    TkSeedSizeMax = cms.double(19.5),
    
    TrackPBin = cms.int32(100),
    TrackPMin = cms.double(0),
    TrackPMax = cms.double(100),

    TrackPtBin = cms.int32(100),
    TrackPtMax = cms.double(30.0),
    TrackPtMin = cms.double(-0.5),
    
    TrackPxBin = cms.int32(100),
    TrackPxMax = cms.double(50.0),
    TrackPxMin = cms.double(-50.0),
    
    TrackPyBin = cms.int32(100),
    TrackPyMax = cms.double(50.0),
    TrackPyMin = cms.double(-50.0),
    
    TrackPzBin = cms.int32(100),
    TrackPzMax = cms.double(50.0),
    TrackPzMin = cms.double(-50.0),
    
    VXBin = cms.int32(200),
    VXMax = cms.double(100.0),
    VXMin = cms.double(-100.0),
    
    VYBin = cms.int32(200),
    VYMax = cms.double(100.0),
    VYMin = cms.double(-100.0),
    
    VZBin = cms.int32(200),
    VZMax = cms.double(200.0),
    VZMin = cms.double(-200.0),
    
    TTRHBuilder = cms.string('WithTrackAngle')
)
