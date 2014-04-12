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
        tag = cms.string('SiStripBadChannel_v1')
    ))
)

process.prod = cms.EDFilter("SiStripBadChannelBuilder",
    printDebug = cms.untracked.bool(True),
    BadComponentList = cms.untracked.VPSet(cms.PSet(
        BadChannelList = cms.vuint32(4, 5, 6, 7, 8, 
            9),
        BadModule = cms.uint32(369120278)
    ), 
        cms.PSet(
            BadChannelList = cms.vuint32(127, 128, 511, 512, 650),
            BadModule = cms.uint32(369140986)
        ), 
        cms.PSet(
            BadChannelList = cms.vuint32(4, 5, 6, 7, 8, 
                9, 511, 512, 650),
            BadModule = cms.uint32(436228654)
        ), 
        cms.PSet(
            BadChannelList = cms.vuint32(4, 5, 6, 7, 8, 
                9, 378, 379, 380, 511, 
                512, 650),
            BadModule = cms.uint32(436294260)
        ), 
        cms.PSet(
            BadChannelList = cms.vuint32(4, 5, 6, 7, 8, 
                9, 127, 128, 378, 379, 
                380, 511, 512, 650),
            BadModule = cms.uint32(470394789)
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


