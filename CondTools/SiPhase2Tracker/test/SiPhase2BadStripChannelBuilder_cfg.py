import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")

process.load("Configuration.StandardSequences.Services_cff")
process.RandomNumberGeneratorService.prod = cms.PSet(
    initialSeed = cms.untracked.uint32(789341),
    engineName = cms.untracked.string('TRandom3')
)

## speciffy detector D49, as the geometry is needed (will take tracker T15)
process.load("Configuration.Geometry.GeometryExtended2026D49_cff")
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
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
    connect = cms.string('sqlite_file:SiStripBadStripPhase2_T15_v0.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadStripRcd'),
        tag = cms.string('SiStripBadStripPhase2_T15')
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
