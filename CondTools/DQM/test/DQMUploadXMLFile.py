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

process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = cms.string('sqlite_file:testXML.db')
process.CondDB.DBParameters.authenticationPath = cms.untracked.string('')
process.CondDB.BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')
#process.CondDBCommon.DBParameters.messageLevel = cms.untracked.int32(3) #3 for high verbosity

process.source = cms.Source("EmptyIOVSource", #needed to EvSetup in order to load data
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDB,
                                          timetype = cms.untracked.string('runnumber'),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('FileBlob'),
                                                                     tag = cms.string('XML_test')
                                                                     )
                                                            ),
                                          logconnect = cms.untracked.string('sqlite_file:XMLFILE_TestLog.db')
                                          )

process.dqmxmlFileTest = cms.EDAnalyzer("DQMXMLFilePopConAnalyzer",
                                        record = cms.string('FileBlob'),
                                        loggingOn = cms.untracked.bool(True), #always True, needs to create the log db
                                        SinceAppendMode = cms.bool(True),
                                        Source = cms.PSet(XMLFile = cms.untracked.string("/cvmfs/cms.cern.ch/slc7_amd64_gcc820/cms/cmssw/CMSSW_11_1_2/src/DQM/SiStripMonitorClient/data/sistrip_qualitytest_config_tier0.xml"),
                                                          firstSince = cms.untracked.uint64(1),
                                                          debug = cms.untracked.bool(True),
                                                          zip = cms.untracked.bool(False)
                                                          )
                                        )

process.p = cms.Path(process.dqmxmlFileTest)
