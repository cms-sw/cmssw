import FWCore.ParameterSet.Config as cms

process = cms.Process("DumpDBToFile")

process.load("CondCore.CondDB.CondDB_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.calibDB = cms.ESSource("PoolDBESSource",
    process.CondDB,
    toGet = cms.VPSet(cms.PSet(
        # TZero
        record = cms.string("DTT0Rcd"),
        tag = cms.string("t0")
    )),
)
process.calibDB.connect = cms.string('sqlite_file:t0.db')

process.dumpToFile = cms.EDAnalyzer("DumpDBToFile",
    dbToDump = cms.untracked.string("TZeroDB"),
    calibFileConfig = cms.untracked.PSet(
        nFields = cms.untracked.int32(7),
        calibConstGranularity = cms.untracked.string('byWire')
    ),
    outputFileName = cms.untracked.string('t0.txt')
)

process.p = cms.Path(process.dumpToFile)
