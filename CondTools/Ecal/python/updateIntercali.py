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
process.CondDBCommon.DBParameters.authenticationPath = '/nfshome0/popcondev/conddb'

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon, 
    logconnect = cms.untracked.string('sqlite_file:log.db'),   
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalIntercalibConstantsRcd'),
        tag = cms.string('EcalIntercalibConstants_test')
    ))
)

process.Test1 = cms.EDAnalyzer("ExTestEcalIntercalibAnalyzer",
    record = cms.string('EcalIntercalibConstantsRcd'),
    loggingOn= cms.untracked.bool(True),
    IsDestDbCheckedInQueryLog=cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
     FileLowField = cms.string('/nfshome0/fra/interCalib_Boff.xml'),
     FileHighField = cms.string('/nfshome0/fra/interCalib_Bon.xml'),
     firstRun = cms.string('98273'),
     lastRun = cms.string('10000000'),
     OnlineDBSID = cms.string('cms_omds_lb'),
     OnlineDBUser = cms.string('cms_ecal_r'),
     OnlineDBPassword = cms.string('*****'),
     LocationSource = cms.string('P5'),
     Location = cms.string('P5_Co'),
     GenTag = cms.string('GLOBAL'),
     RunType = cms.string('COSMICS')
    )                            
)

process.p = cms.Path(process.Test1)
