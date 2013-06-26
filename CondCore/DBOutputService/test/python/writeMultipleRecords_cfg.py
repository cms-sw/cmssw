import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string("sqlite_file:test.db")

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('a'),
        tag = cms.string('ped_tag')
    ), cms.PSet(
        record = cms.string('b'),
        tag = cms.string('noise_tag')
    ))
)

process.mytest = cms.EDAnalyzer("writeMultipleRecords",
                  PedCallerName=cms.string('a'),
                  StripCallerName=cms.string('b')
)

process.p = cms.Path(process.mytest)



