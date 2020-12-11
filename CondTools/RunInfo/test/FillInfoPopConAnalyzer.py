import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:fillinfo_pop_test.db'
process.CondDBCommon.DBParameters.authenticationPath = '.'
process.CondDBCommon.DBParameters.messageLevel=cms.untracked.int32(1)

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('INFO')
    )
)

process.source = cms.Source("EmptyIOVSource",
                            lastValue = cms.uint64(1),
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          logconnect = cms.untracked.string('sqlite_file:logfillinfo_pop_test.db'),
                                          timetype = cms.untracked.string('timestamp'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('FillInfoRcd'),
                                                                     tag = cms.string('fillinfo_test')
                                                                     )
                                                            )
                                          )

process.Test1 = cms.EDAnalyzer("FillInfoPopConAnalyzer",
                               SinceAppendMode = cms.bool(True),
                               record = cms.string('FillInfoRcd'),
                               name = cms.untracked.string('FillInfo'),
                               Source = cms.PSet(fill = cms.untracked.uint32(902),
                                                 connectionString = cms.untracked.string("oracle://ora_db/ora_schema"),
                                                 authenticationPath =  cms.untracked.string(".")
                                                 ),
                               loggingOn = cms.untracked.bool(True),
                               IsDestDbCheckedInQueryLog = cms.untracked.bool(False)
                               )

process.p = cms.Path(process.Test1)



