import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:keys.db'

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('hash'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('uniqueCrappyName'),
        tag = cms.string('KeyTest')
    ))
)

process.mytest = cms.EDAnalyzer("writeKeyed")

process.p = cms.Path(process.mytest)

