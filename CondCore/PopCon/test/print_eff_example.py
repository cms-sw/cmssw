import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:pop_test.db'

process.eff = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    RefreshEachRun=cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('ExEfficiencyRcd'),
        tag = cms.string('Example_tag1')
    ))
)

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(310),
    lastValue = cms.uint64(1000),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(100)
)

process.prod = cms.EDAnalyzer("EfficiencyByLabelAnalyzer")

process.p = cms.Path(process.prod)

