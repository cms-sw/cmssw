import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.a = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('ped_tag')
    )),
    connect = cms.string('frontier://FrontierPrep/CMS_CONDITIONS')
)

process.b = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('anotherPedestalsRcd'),
        tag = cms.string('mytest')
    )),
                         connect = cms.string('frontier://FrontierPrep/CMS_CONDITIONS')
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(43),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(31),
    interval = cms.uint64(1)
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        data = cms.vstring('Pedestals')
    ), cms.PSet(
        record = cms.string('anotherPedestalsRcd'),
        data = cms.vstring('Pedestals')
    )),
    verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.get)


