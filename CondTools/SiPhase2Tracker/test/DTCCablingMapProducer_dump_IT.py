import FWCore.ParameterSet.Config as cms

process = cms.Process("DTCCablingMapDump")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
    cout = cms.untracked.PSet(enable = cms.untracked.bool(True),
                              threshold = cms.untracked.string('INFO')),
)

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = 'sqlite_file:OTandITDTCCablingMap.db'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    DumpStat = cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerDetToDTCELinkCablingMapRcd'),
        tag    = cms.string("DTCCablingMapProducerUserRun"),
    )),
)

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(1),
    interval   = cms.uint64(1),
)

process.reader = cms.EDAnalyzer("DTCCablingMapTestReader")
process.path = cms.Path(process.reader)
