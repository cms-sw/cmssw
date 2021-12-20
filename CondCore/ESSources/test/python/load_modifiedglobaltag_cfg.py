import FWCore.ParameterSet.Config as cms
from Configuration.AlCa.autoCond import autoCond

process = cms.Process("TEST")
process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
process.CondDB.DBParameters.messageLevel = 0

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    globaltag = cms.string(autoCond['run2_data']),
    toGet = cms.VPSet(cms.PSet(
            connect=cms.string('frontier://FrontierPrep/CMS_CONDITIONS'),
            record = cms.string('anotherPedestalsRcd'),
            tag = cms.string('mytest'),
            label = cms.untracked.string('')
            ) )
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(43),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(31),
    interval = cms.uint64(1)
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('anotherPedestalsRcd'),
        data = cms.vstring('Pedestals')
    ), cms.PSet(
        record = cms.string('ESPedestalsRcd'),
        data = cms.vstring('ESPedestals')
    )),
    verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.get)



