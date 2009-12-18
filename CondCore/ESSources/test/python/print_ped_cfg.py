import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:test.db'


process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('mytest')
    )),
)

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(16),
    numberEventsInRun = cms.untracked.uint32(1)
)

process.maxEvents=cms.untracked.PSet(input=cms.untracked.int32(5))
process.prod = cms.EDAnalyzer("PedestalsAnalyzer")

process.p = cms.Path(process.prod)


