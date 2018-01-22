import FWCore.ParameterSet.Config as cms
process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
#process.CondDBCommon.connect = 'sqlite_file:test_output.db'
#process.CondDBCommon.DBParameters.authenticationPath = '.'
process.CondDBCommon.connect = 'oracle://cms_orcoff_prep/CMS_CONDITIONS'
process.CondDBCommon.DBParameters.authenticationPath =  '/afs/cern.ch/user/a/anoolkar/private'
process.CondDBCommon.DBParameters.messageLevel=cms.untracked.int32(3)

process.MessageLogger = cms.Service("MessageLogger",
                                    cout = cms.untracked.PSet(threshold = cms.untracked.string('INFO')),
                                    destinations = cms.untracked.vstring('cout')
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
                               Source = cms.PSet(fill = cms.untracked.uint32(6303),
                                   firstFill = cms.untracked.uint32( 6300 ),
                                   lastFill = cms.untracked.uint32( 6300 ),
                                   connectionString = cms.untracked.string("oracle://cms_orcon_adg/CMS_RUNTIME_LOGGER"),
                                   DIPSchema = cms.untracked.string("CMS_BEAM_COND"),
                                   authenticationPath =  cms.untracked.string("/afs/cern.ch/user/a/anoolkar/private"),
                                   debug=cms.untracked.bool(True)
                                                 ),
                               loggingOn = cms.untracked.bool(True),
                               IsDestDbCheckedInQueryLog = cms.untracked.bool(False)
                               )

process.p = cms.Path(process.Test1)
