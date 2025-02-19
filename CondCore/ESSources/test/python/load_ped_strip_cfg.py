import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:blob.db'
    #'oracle://devdb10/cms_xiezhen_dev'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('mySiStripNoisesRcd'),
        tag = cms.string('noise_tag')
    )),
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(10),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('mySiStripNoisesRcd'),
        data = cms.vstring('mySiStripNoises')
    )),
    verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.get)


