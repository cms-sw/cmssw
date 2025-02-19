import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")
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

process.prod = cms.EDFilter("SiStripBadStripFromASCIIFile",
    Record = cms.string('SiStripBadStrip'),
    printDebug = cms.untracked.bool(True),
    IOVMode = cms.string('Run'),
    SinceAppendMode = cms.bool(True),
    doStoreOnDB = cms.bool(True),
    file = cms.untracked.FileInPath('CalibTracker/SiStripQuality/data/DefectsFromConstructionDB.dat')
)

process.p = cms.Path(process.prod)


