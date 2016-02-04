import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.MessageLogger=cms.Service("MessageLogger",
                                  cout=cms.untracked.PSet(threshold=cms.untracked.string('INFO')),
                                  destinations=cms.untracked.vstring("cout")
                                  )

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = cms.string('sqlite_file:testXML.db')
process.CondDBCommon.DBParameters.authenticationPath = cms.untracked.string('')
process.CondDBCommon.BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService')

#process.CondDBCommon.DBParameters.messageLevel = cms.untracked.int32(3) #3 for high verbosity

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
                                                                     tag = cms.string('XML_pixels_1')
                                                                     )
                                                            ),
                                          logconnect = cms.untracked.string('sqlite_file:XMLFILE_TestLog.db')                                     
                                          )

process.dqmxmlFileTest = cms.EDAnalyzer("DQMXMLFilePopConAnalyzer",
                                        record = cms.string('FileBlob'),
                                        loggingOn = cms.untracked.bool(True), #always True, needs to create the log db
                                        SinceAppendMode = cms.bool(True),
                                        Source = cms.PSet(XMLFile = cms.untracked.string("sipixel_qualitytest_config.xml"),
    #"/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/CMSSW_3_3_0/src/DQM/SiStripMonitorClient/data/sistrip_qualitytest_config_tier0.xml"),
    #"/afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/cmssw/CMSSW_3_3_0/src/DQM/SiPixelMonitorClient/test/sipixel_qualitytest_config.xml"),
                                                          firstSince = cms.untracked.uint64(124275), #1, 43434, 46335, 51493, 51500
                                                          debug = cms.untracked.bool(False),
                                                          zip = cms.untracked.bool(False)
                                                          )
                                        )

process.p = cms.Path(process.dqmxmlFileTest)
