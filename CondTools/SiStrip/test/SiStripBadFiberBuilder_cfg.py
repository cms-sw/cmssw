import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")
process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripBadStrip'),
        tag = cms.string('SiStripBadFiber_v1')
    ))
)

process.prod = cms.EDFilter("SiStripBadFiberBuilder",
    printDebug = cms.untracked.bool(True),
    BadComponentList = cms.untracked.VPSet(cms.PSet(
        BadModule = cms.uint32(369120278),
        BadApvList = cms.vuint32(0, 1, 2, 3, 5)
    ), 
        cms.PSet(
            BadModule = cms.uint32(369140986),
            BadApvList = cms.vuint32(4, 5)
        ), 
        cms.PSet(
            BadModule = cms.uint32(436228654),
            BadApvList = cms.vuint32(0, 1, 2, 3)
        ), 
        cms.PSet(
            BadModule = cms.uint32(436294260),
            BadApvList = cms.vuint32(0, 2, 4)
        ), 
        cms.PSet(
            BadModule = cms.uint32(470394789),
            BadApvList = cms.vuint32(2, 3, 4, 5)
        )),
    IOVMode = cms.string('Run'),
    Record = cms.string('SiStripBadStrip'),
    doStoreOnDB = cms.bool(True),
    file = cms.untracked.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
    SinceAppendMode = cms.bool(True)
)

process.print = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.print)


