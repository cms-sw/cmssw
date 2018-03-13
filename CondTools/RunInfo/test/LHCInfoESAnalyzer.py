import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:lhcinfo_pop_test.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)


process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('timestamp'),
    firstValue = cms.uint64(6477060826480649336),
    lastValue = cms.uint64(6477060826480649336),
    interval = cms.uint64( 3600 << 32 )
)


process.rn = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    timetype = cms.string('timestamp'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('LHCInfoRcd'),
        tag = cms.string('lhcinfo_test')
    ))
)


process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('LHCInfoRcd'),
        data = cms.vstring('lhcinfo_test')
    )),
    verbose = cms.untracked.bool(True)
)

process.prod = cms.EDAnalyzer("LHCInfoESAnalyzer")

process.asciiprint = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.asciiprint)
