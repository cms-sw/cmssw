import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string("sqlite_file:test.db")
process.CondDBCommon.DBParameters.messageLevel = 3

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('mytest')
    ))
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(18),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(16),
    interval = cms.uint64(1)
)

process.prod = cms.EDAnalyzer("PedestalsAnalyzer")

process.p = cms.Path(process.prod)


# foo bar baz
# TFGT8qcWdayFo
# geiV5ll1DD5AK
