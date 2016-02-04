import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'oracle://cms_orcoff_prod/CMS_COND_21X_RUN_INFO'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(66394),
    lastValue = cms.uint64(66400),
    interval = cms.uint64(1)
)



process.rn = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RunInfoRcd'),
        tag = cms.string('runinfo_test')
    ))
)


process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('RunInfoRcd'),
        data = cms.vstring('runinfo_test')
    )),
    verbose = cms.untracked.bool(True)
)

process.prod = cms.EDAnalyzer("RunInfoESAnalyzer")

process.asciiprint = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.asciiprint)
