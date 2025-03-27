import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

CondDBSetup = cms.PSet(DBParameters = cms.PSet(messageLevel = cms.untracked.int32(1)))

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    CondDBSetup,
    connect = cms.string(''),
    toGet = cms.VPSet(
        cms.PSet(
            connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS'),
            record = cms.string('DTTtrigRcd'),
            tag = cms.string('DTTtrig_STARTUP_V01_mc'),
            label = cms.untracked.string('t2')
        ), 
        cms.PSet(
            connect = cms.string('frontier://FrontierPrep/CMS_CONDITIONS'),
            record = cms.string('DTTtrigRcd'),
            tag = cms.string('DTTtrig_IDEAL_V02_mc'),
            label = cms.untracked.string('t1')
        )
     )
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(3),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('DTTtrigRcd'),
        data = cms.vstring('DTTtrig/t1', 'DTTtrig/t2')
    )),
    verbose = cms.untracked.bool(True)
)

process.p = cms.Path(process.get)



