import FWCore.ParameterSet.Config as cms

process = cms.Process('XMLFILERETRIEVER')

# setting the Global Tag
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.connect = "sqlite_file:./tagDB.db" 
process.GlobalTag.globaltag = 'MY_GT::All' 
process.GlobalTag.DumpStat = cms.untracked.bool(True)

# import of standard configurations
process.load("DQMServices.Core.DQMStore_cfg")
process.DQMStore.verbose   = cms.untracked.int32(1)
process.DQMStore.verboseQT = cms.untracked.int32(0)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

# Input source
from CondCore.DBCommon.CondDBSetup_cfi import *

process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )
                                      
#process.XmlRetrieval_2 = cms.ESSource("PoolDBESSource",
#                                      CondDBSetup,
#                                      connect = cms.string('sqlite_file:./testXML.db'),
#                                      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#                                      messageLevel = cms.untracked.int32(1),
#                                      timetype = cms.string('runnumber'),
#                                      toGet = cms.VPSet(cms.PSet(record = cms.string('DQMXMLFileRcd'),
#                                                                 tag = cms.string('XML_pixels_1'),
#                                                                 label=cms.untracked.string('XML2_mio')
#                                                                 )
#                                                        )
#                                      )

#process.XmlRetrieval_1 = cms.ESSource("PoolDBESSource",
#                                      CondDBSetup,
#                                      connect = cms.string('sqlite_file:./testXML.db'),
#                                      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#                                      messageLevel = cms.untracked.int32(1),
#                                      timetype = cms.string('runnumber'),
#                                      toGet = cms.VPSet(cms.PSet(record = cms.string('DQMXMLFileRcd'),
#                                                                 tag = cms.string('XML1'),
#                                                                 label=cms.untracked.string('XML1_mio')
#                                                                 )
#                                                        )
#                                      )


process.load('CondTools/DQM/DQMXMLFileEventSetupAnalyzer_cfi')
process.dqmXMLFileGetter.labelToGet = cms.string('fuffa')

process.path = cms.Path(process.dqmXMLFileGetter)

#process.path = cms.Path()
