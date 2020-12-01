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
process.CondDBCommon.connect = cms.string('sqlite_file:ROOTFILE_Test.db')
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
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('FileBlob'),
                                                                     tag = cms.string('ROOTFILE_Test') 
                                                                     )
                                                            ),
                                          logconnect = cms.untracked.string('sqlite_file:ROOTFILE_TestLog.db')                                     
                                          )

process.dqmReferenceHistogramRootFileTest = cms.EDAnalyzer("DQMReferenceHistogramRootFilePopConAnalyzer",
                                           record = cms.string('FileBlob'),
                                           loggingOn = cms.untracked.bool(True), #always True, needs to create the log db
                                           SinceAppendMode = cms.bool(True),
                                           Source = cms.PSet(ROOTFile = cms.untracked.string("salvo.root"),
                                                             firstSince = cms.untracked.uint64(1), #1, 43434, 46335, 51493, 51500
                                                             debug = cms.untracked.bool(True)
                                                             )
                                           )

process.p = cms.Path(process.dqmReferenceHistogramRootFileTest)
