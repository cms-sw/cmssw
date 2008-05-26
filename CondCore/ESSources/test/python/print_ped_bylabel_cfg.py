import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.a = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('mytest'),
        label = cms.untracked.string('lab3d')
    ), cms.PSet(
        record = cms.string('PedestalsRcd'),
        tag = cms.string('pedtag'),
        label = cms.untracked.string('PEDPED')
    ), cms.PSet(
        record = cms.string('anotherPedestalsRcd'),
        tag = cms.string('anothermytest'),
        label = cms.untracked.string('Three')
    )),
    connect = cms.string('sqlite_file:test.db')
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(3),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.prod = cms.EDAnalyzer("PedestalsByLabelAnalyzer")

process.p = cms.Path(process.prod)


