import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('mytest')
    ), cms.PSet(
        record = cms.string('anotherPedestalsRcd'),
        tag = cms.string('anothermytest')
    )),
    connect = cms.string('sqlite_file:test.db')
)

process.source = cms.Source("EmptyIOVSource",
    lastRun = cms.untracked.uint32(3),
    timetype = cms.string('runnumber'),
    firstRun = cms.untracked.uint32(1),
    interval = cms.uint32(1)
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


