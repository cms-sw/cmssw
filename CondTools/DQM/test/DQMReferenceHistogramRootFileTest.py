import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger=cms.Service("MessageLogger",
                                  cout=cms.untracked.PSet(threshold=cms.untracked.string('INFO')),
                                  destinations=cms.untracked.vstring("cout")
                                  )

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string('sqlite_file:DQMReferenceHistogramTest.db')
process.CondDBCommon.BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
process.CondDBCommon.DBParameters.messageLevel = cms.untracked.int32(1) #3 for high verbosity

process.source = cms.Source("EmptyIOVSource", #needed to EvSetup in order to load data
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBCommon,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('DQMReferenceHistogramRootFileRcd'),
                                                                     tag = cms.string('ROOTFILE_DQM_Test') 
                                                                     )
                                                            ),
                                          logconnect = cms.untracked.string('sqlite_file:DQMReferenceHistogramTestLog.db')                                     
                                          )

process.dqmReferenceHistogramRootFileTest = cms.EDAnalyzer("DQMReferenceHistogramRootFilePopConAnalyzer",
                                           record = cms.string('DQMReferenceHistogramRootFileRcd'),
                                           loggingOn = cms.untracked.bool(True), #always True, needs to create the log db
                                           SinceAppendMode = cms.bool(True),
                                           Source = cms.PSet(ROOTFile = cms.untracked.string("file.root"),
                                                             firstSince = cms.untracked.uint64(1), #1, 43434, 46335, 51493, 51500
                                                             debug = cms.untracked.bool(True)
                                                             )
                                           )

process.p = cms.Path(process.dqmReferenceHistogramRootFileTest)
