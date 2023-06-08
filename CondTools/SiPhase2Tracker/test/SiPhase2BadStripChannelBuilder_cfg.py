import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")

process.load("Configuration.StandardSequences.Services_cff")
process.RandomNumberGeneratorService.prod = cms.PSet(
    initialSeed = cms.untracked.uint32(789341),
    engineName = cms.untracked.string('TRandom3')
)

## specify detector D88, as the geometry is needed (will take tracker T24)
process.load("Configuration.Geometry.GeometryExtended2026D88_cff")
process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

###################################################################
# Messages
###################################################################
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.enable = False
process.MessageLogger.SiPhase2BadStripChannelBuilder=dict()
process.MessageLogger.SiStripBadStrip=dict()
process.MessageLogger.cout = cms.untracked.PSet(
    enable    = cms.untracked.bool(True),
    enableStatistics = cms.untracked.bool(True),
    threshold = cms.untracked.string("INFO"),
    default   = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(-1),
                                   reportEvery = cms.untracked.int32(1000)
                                   ),
    SiPhase2BadStripChannelBuilder = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
    SiStripBadStrip = cms.untracked.PSet( limit = cms.untracked.int32(-1)),
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:SiStripBadStripPhase2_T21_v0.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadStripRcd'),
        tag = cms.string('SiStripBadStripPhase2_T21')
    ))
)

process.prod = cms.EDAnalyzer("SiPhase2BadStripChannelBuilder",                    
                              Record = cms.string('SiStripBadStripRcd'),
                              SinceAppendMode = cms.bool(True),
                              IOVMode = cms.string('Run'),
                              printDebug = cms.untracked.bool(False),
                              doStoreOnDB = cms.bool(True),
                              #popConAlgo = cms.uint32(1), #NAIVE
                              popConAlgo = cms.uint32(2), #RANDOM
                              badComponentsFraction = cms.double(0.01)  #1% of bad strips
                              #badComponentsFraction = cms.double(0.05)  #5% of bad strips
                              #badComponentsFraction = cms.double(0.1)   #10% of bad strips
                              )

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
#process.ep = cms.EndPath(process.print)
