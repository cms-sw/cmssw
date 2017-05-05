import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('MuonSystemAgingRcd'),
        tag = cms.string('MuonSystemAging_test')
    )),
    connect = cms.string('sqlite_file:MuonSystemAging.db')
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(3),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('MuonSystemAgingRcd'),
        data = cms.vstring('MuonSystemAging')
    )),
    verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.get)
