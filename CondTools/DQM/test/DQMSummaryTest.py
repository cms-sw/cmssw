import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_30X_DQM_SUMMARY')
process.CondDBCommon.connect = cms.string('sqlite_file:DQMSummaryTest.db')
#process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb') #authentication when using ORACLE
process.CondDBCommon.DBParameters.messageLevel = cms.untracked.int32(1) #3 for high verbosity

process.source = cms.Source("EmptyIOVSource", #needed to EvSetup in order to load data
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'), #IOV: 'runnumber'-> number of the run, 'timestamp'-> microseconds starting from 1/1/1970
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('DQMSummary'),
        tag = cms.string('DQMSummaryTest') 
         )),
    logconnect= cms.untracked.string('sqlite_file:DQMSummaryLogTest.db')                                     
)

process.dqmSummaryTest = cms.EDAnalyzer("DQMSummaryPopConAnalyzer",
    record = cms.string('DQMSummary'),
    loggingOn = cms.untracked.bool(True), #always True, needs to create the log db
    SinceAppendMode = cms.bool(True),
    Source = cms.PSet(
    firstSince = cms.untracked.uint64(43434) #1, 43434, 46335, 51493, 51500
    )                            
)

process.p = cms.Path(process.dqmSummaryTest)
