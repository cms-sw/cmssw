import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string("sqlite_file:tagDB.db")
process.CondDBCommon.DBParameters.messageLevel = 0

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    globaltag = cms.untracked.string('MYTREE1::All')
)

process.source = cms.Source("EmptyIOVSource",
    lastRun = cms.untracked.uint32(3),
    timetype = cms.string('runnumber'),
    firstRun = cms.untracked.uint32(1),
    interval = cms.uint32(1)
)

process.get = cms.EDFilter("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('anotherPedestalsRcd'),
        data = cms.vstring('Pedestals')
    ), cms.PSet(
        record = cms.string('PedestalsRcd'),
        data = cms.vstring('Pedestals/lab3d', 'Pedestals/lab2')
    )),
    verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.get)



