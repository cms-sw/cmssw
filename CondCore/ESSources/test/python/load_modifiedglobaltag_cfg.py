import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string("sqlite_file:tagDB.db")
process.CondDBCommon.DBParameters.messageLevel = 0

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDBCommon,
    globaltag = cms.string('MYTREE1::All'),
    toGet = cms.VPSet(cms.PSet(
            connect=cms.untracked.string('sqlite_file:extradata.db'),
            record = cms.string('anotherPedestalsRcd'),
            tag = cms.untracked.string('anothertag'),
            label = cms.untracked.string('')
            ) )
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(3),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
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



