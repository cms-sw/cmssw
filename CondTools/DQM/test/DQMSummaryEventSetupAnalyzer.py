import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:DQMSummaryTest.db'
#process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_COND_30X_DQM_SUMMARY'
#process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb/'

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(43434),
    lastValue = cms.uint64(43434),
    interval = cms.uint64(1)
)

process.rn = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DQMSummaryRcd'),
        tag = cms.string('DQMSummaryTest')
    ))
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DQMSummaryRcd'),
        data = cms.vstring('DQMSummaryTest')
    )),
    verbose = cms.untracked.bool(False)
)

process.prod = cms.EDAnalyzer("DQMSummaryEventSetupAnalyzer")

process.asciiprint = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod)
process.ep = cms.EndPath(process.asciiprint)
