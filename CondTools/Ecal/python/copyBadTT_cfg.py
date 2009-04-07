import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptyIOVSource",
    lastValue = cms.uint64(100000000000),
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(100000000000),
    interval = cms.uint64(1)
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")

process.CondDBCommon.connect = 'sqlite_file:DB.db'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon, 
    logconnect = cms.untracked.string('sqlite_file:log.db'),   
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalTPGTowerStatusRcd'),
#        tag = cms.string('EcalTPGTowerStatus_craft')
       tag = cms.string('EcalTPGTowerStatus_TPGTrivial_config')
    ))
)

process.Test1 = cms.EDAnalyzer("ExTestEcalTPGBadTTAnalyzer",
    record = cms.string('EcalTPGTowerStatusRcd'),
    loggingOn= cms.untracked.bool(True),
    IsDestDbCheckedInQueryLog=cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
     firstRun = cms.string('13'),
     lastRun = cms.string('10000000'),
     OnlineDBSID = cms.string('ecalh4db'),
     OnlineDBUser = cms.string('test01'),
     OnlineDBPassword = cms.string('oratest01'),
     LocationSource = cms.string('P5'),
     Location = cms.string('H4'),
     GenTag = cms.string('default')
    )                            
)

process.p = cms.Path(process.Test1)
