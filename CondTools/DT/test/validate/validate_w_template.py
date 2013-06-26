import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:test.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    logconnect = cms.untracked.string('sqlite_file:log.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('CLASSRcd'),
        tag = cms.string('FNAME_test')
    ))
)

process.source = cms.Source("EmptyIOVSource",
    timetype   = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue  = cms.uint64(2),
    interval   = cms.uint64(1)
)

process.FNAME = cms.EDAnalyzer("VNAMEValidateDBWrite",
    name = cms.untracked.string('CLASS'),
    record = cms.string('CLASSRcd'),
    Source = cms.PSet(
        version = cms.string('FNAME_test'),
        outFile = cms.string('FNAMEDump.txt'),
        logFile = cms.string('FNAMEValidate.log'),
        debug = cms.bool(False),
        firstRun = cms.uint32(1),
         lastRun = cms.uint32(5)
    ),
    SinceAppendMode = cms.bool(True)
)

process.p = cms.Path(process.FNAME)

