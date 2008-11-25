# The following comments couldn't be translated into the new config version:

# All/OuterSurface/InnerSurface/ImpactPoint/default(track)
#

import FWCore.ParameterSet.Config as cms

TrackMon = cms.EDFilter("TrackingMonitor",
    TrackProducer = cms.InputTag("generalTracks"),
    AlgoName = cms.string('GenTk'),
                        
    OutputMEsInRootFile = cms.bool(False),
    OutputFileName = cms.string('MonitorTrack.root'),

    FolderName = cms.string('Track/GlobalParameters'),

    MeasurementState = cms.string('default'),

    doTrackerSpecific = cms.bool(False),
                        
    TkSizeBin = cms.int32(500),
    TkSizeMin = cms.double(-0.5),
    TkSizeMax = cms.double(499.5),

    TrackPtBin = cms.int32(1000),
    TrackPtMin = cms.double(0),
    TrackPtMax = cms.double(1000),

    ptErrBin = cms.int32(100),
    ptErrMin = cms.double(0.0),
    ptErrMax = cms.double(10.0),

    D0Max = cms.double(0.5),
    D0Min = cms.double(-0.5),                        
    D0Bin = cms.int32(100),
                        
    etaErrBin = cms.int32(100),
    etaErrMin = cms.double(0.0),
    etaErrMax = cms.double(0.5),

    VXBin = cms.int32(20),
    VXMin = cms.double(-20.0),
    VXMax = cms.double(20.0),

    RecHitBin = cms.int32(100),
    RecHitMin = cms.double(-0.5),
    RecHitMax = cms.double(99.5),

    RecLostBin = cms.int32(20),
    RecLostMin = cms.double(-0.5),
    RecLostMax = cms.double(19.5),

    RecLayBin = cms.int32(100),
    RecLayMin = cms.double(-0.5),
    RecLayMax = cms.double(99.5),
                        
    Chi2Max = cms.double(500.0),
    Chi2Bin = cms.int32(100),
    Chi2Min = cms.double(-0.5),

    VYBin = cms.int32(20),
    VYMin = cms.double(-20.0),
    VYMax = cms.double(20.0),

    VZBin = cms.int32(50),
    VZMin = cms.double(-100.0),
    VZMax = cms.double(100.0),

    TrackPzBin = cms.int32(1000),
    TrackPzMin = cms.double(-500.0),
    TrackPzMax = cms.double(500.0),
                        
    ThetaBin = cms.int32(100),
    ThetaMin = cms.double(0.0),
    ThetaMax = cms.double(3.2),

    EtaBin = cms.int32(100),
    EtaMin = cms.double(-4.0),
    EtaMax = cms.double(4.0),

    PhiBin = cms.int32(100),
    PhiMin = cms.double(-3.2),
    PhiMax = cms.double(3.2),

    phiErrBin = cms.int32(100),
    phiErrMin = cms.double(0.0),
    phiErrMax = cms.double(1.0),

    TrackPxBin = cms.int32(1000),
    TrackPxMin = cms.double(-500.0),
    TrackPxMax = cms.double(500.0),

    TrackPyBin = cms.int32(1000),
    TrackPyMin = cms.double(-500.0),
    TrackPyMax = cms.double(500.0),

    pErrBin = cms.int32(100),
    pErrMin = cms.double(0.0),
    pErrMax = cms.double(10.0),

    pxErrBin = cms.int32(100),
    pxErrMax = cms.double(10.0),
    pxErrMin = cms.double(0.0),

    pyErrBin = cms.int32(100),                        
    pyErrMin = cms.double(0.0),
    pyErrMax = cms.double(10.0),

    pzErrBin = cms.int32(100),
    pzErrMin = cms.double(0.0),
    pzErrMax = cms.double(10.0)

)


