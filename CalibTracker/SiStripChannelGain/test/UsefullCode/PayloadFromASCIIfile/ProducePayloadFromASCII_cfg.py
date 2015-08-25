import FWCore.ParameterSet.Config as cms

process = cms.Process("APVGAIN")

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet( threshold = cms.untracked.string('INFO')  ), #use ERROR for less printouts
    destinations = cms.untracked.vstring('cout')
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'GR_P_V53', '')

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    interval   = cms.uint64(1),
    firstValue = cms.uint64(251522),
    lastValue  = cms.uint64(251612)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:Gains_Sqlite.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        tag = cms.string('SiStripGainFromParticles')
    ))
)


process.SiStripCalibFromAscii = cms.EDAnalyzer("SiStripGainFromAsciiFile",
    InputFileName       = cms.string('Gains_ASCII.txt'),
    referenceValue      = cms.double(1.0),

    SinceAppendMode     = cms.bool(True),
    TimeFromEndRun      = cms.untracked.bool(True),
    IOVMode             = cms.string('Job'),
    Record              = cms.string('SiStripApvGainRcd'),
    doStoreOnDB         = cms.bool(True)
)



process.p = cms.Path(process.SiStripCalibFromAscii) 
