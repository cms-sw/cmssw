# The following comments couldn't be translated into the new config version:

#

import FWCore.ParameterSet.Config as cms

MonitorTrackTKCosmicMuons = cms.EDFilter("TrackingMonitor",
    OutputMEsInRootFile = cms.bool(False),
    TrackPxBin = cms.int32(100),
    RecHitBin = cms.int32(22),
    TrackPzMin = cms.double(-50.0),
    Chi2Max = cms.double(500.0),
    Chi2Bin = cms.int32(100),
    TrackPzBin = cms.int32(100),
    TrackPxMax = cms.double(50.0),
    TrackPzMax = cms.double(50.0),
    ThetaBin = cms.int32(100),
    RecHitMin = cms.double(0.0),
    MTCCData = cms.bool(False),
    OutputFileName = cms.string('monitortrackparameters_tkmuons.root'),
    TrackPxMin = cms.double(-50.0),
    EtaMin = cms.double(-4.0),
    EtaMax = cms.double(4.0),
    Chi2Min = cms.double(-0.5),
    ThetaMin = cms.double(0.0),
    PhiMin = cms.double(-3.2),
    TrackPtMax = cms.double(30.0),
    RecHitMax = cms.double(25.0),
    TrackPyBin = cms.int32(100),
    TkSizeMin = cms.double(0.0),
    ThetaMax = cms.double(3.2),
    EtaBin = cms.int32(100),
    #
    FolderName = cms.string('Muons/TKTrack'),
    TkSizeBin = cms.int32(500),
    AlgoName = cms.string('ctf'),
    TrackPyMin = cms.double(-50.0),
    TrackProducer = cms.string('ctfWithMaterialTracksP5'),
    TkSizeMax = cms.double(500.0),
    TrackPtBin = cms.int32(100),
    TrackPyMax = cms.double(50.0),
    TrackLabel = cms.string(''),
    PhiBin = cms.int32(100),
    PhiMax = cms.double(3.2),
    TrackPtMin = cms.double(-0.5)
)


