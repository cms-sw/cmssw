import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:fillinfo_pop_test.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('timestamp'),
    firstValue = cms.uint64(5412240172618088448),
    lastValue = cms.uint64(5412267527264796672),
    interval = cms.uint64( 3600 << 32 )
)



process.rn = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    timetype = cms.string('timestamp'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('FillInfoRcd'),
        tag = cms.string('fillinfo_test')
    ))
)


process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('FillInfoRcd'),
        data = cms.vstring('fillinfo_test')
    )),
    verbose = cms.untracked.bool(True)
)

process.prod = cms.EDAnalyzer("FillInfoESAnalyzer")

process.asciiprint = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.asciiprint)
