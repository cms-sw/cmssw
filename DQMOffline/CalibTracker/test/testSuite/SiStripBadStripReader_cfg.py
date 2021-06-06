import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
    lastRun = cms.untracked.uint32(1),
    timetype = cms.string('runnumber'),
    firstRun = cms.untracked.uint32(1),
    interval = cms.uint32(1)
)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.Timing = cms.Service("Timing")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string("SiStripBadStripRcd"),
        tag = cms.string("SiStripBadStrip_test1")
#        record = cms.string('SiStripBadStripRcd'),
#        tag = cms.string('SiStripBadChannel_v1')
    )),
    connect = cms.string('sqlite_file:dbfile.db')
)

process.prod = cms.EDFilter("SiStripBadStripReader",
    #printDebug = cms.untracked.bool(True)
)

#process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
#process.ep = cms.EndPath(process.print)


