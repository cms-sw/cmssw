# The following comments couldn't be translated into the new config version:

#

import FWCore.ParameterSet.Config as cms

# MonitorTrackGlobal
MonitorTrackSTACosmicMuons = cms.EDFilter("MonitorTrackGlobal",
    OutputMEsInRootFile = cms.bool(True),
    TrackPxBin = cms.int32(100),
    RecHitBin = cms.int32(90),
    TrackPzMin = cms.double(-50.0),
    Chi2Max = cms.double(500.0),
    Chi2Bin = cms.int32(100),
    TrackPzBin = cms.int32(100),
    TrackPxMax = cms.double(50.0),
    TrackPzMax = cms.double(50.0),
    ThetaBin = cms.int32(100),
    RecHitMin = cms.double(0.0),
    #
    MTCCData = cms.bool(False),
    OutputFileName = cms.string('monitortrackparameters_stamuons.root'),
    TrackPxMin = cms.double(-50.0),
    EtaMin = cms.double(-3.0),
    EtaMax = cms.double(3.0),
    Chi2Min = cms.double(-0.5),
    ThetaMin = cms.double(0.0),
    PhiMin = cms.double(-3.2),
    TrackPtMax = cms.double(19.5),
    RecHitMax = cms.double(90.0),
    TrackPyBin = cms.int32(100),
    TkSizeMin = cms.double(-0.5),
    ThetaMax = cms.double(3.2),
    EtaBin = cms.int32(100),
    TkSizeBin = cms.int32(11),
    AlgoName = cms.string('sta'),
    TrackPyMin = cms.double(-50.0),
    TrackProducer = cms.string('cosmicMuons'),
    TkSizeMax = cms.double(10.5),
    TrackPtBin = cms.int32(20),
    TrackPyMax = cms.double(50.0),
    TrackLabel = cms.string(''),
    PhiBin = cms.int32(100),
    PhiMax = cms.double(3.2),
    TrackPtMin = cms.double(-0.5)
)


