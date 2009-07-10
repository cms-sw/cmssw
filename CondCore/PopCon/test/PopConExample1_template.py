import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger=cms.Service("MessageLogger",
                              destinations=cms.untracked.vstring("cout"),
                              cout=cms.untracked.PSet(
#                              treshold=cms.untracked.string("INFO")
                              )
)

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string('sqlite_file:pop_test.db')
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')

process.source = cms.Source("EmptyIOVSource",
    timetype = cms.string('runnumber'),
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    interval = cms.uint64(1)
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    withWrapper = cms.untracked.bool(False),
    outOfOrder = cms.untracked.bool(False),
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('CSCPedestalsRcd'),
        tag = cms.string('Example_tag')
         )),
    logconnect= cms.untracked.string('sqlite_file:log.db')                                     
)

process.mytest = cms.EDAnalyzer("ExPopConAnalyzer",
    record = cms.string('CSCPedestalsRcd'),
    loggingOn= cms.untracked.bool(True),
    SinceAppendMode=cms.bool(True),
    Source=cms.PSet(
    firstSince=cms.untracked.int64(xxx),
    number=cms.untracked.int64(yyy)
    )                            
)

process.p = cms.Path(process.mytest)
