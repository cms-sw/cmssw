import FWCore.ParameterSet.Config as cms

process = cms.Process('XMLFILERETRIEVER')

####################################################################
# Get the GlogalTag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data_promptlike', '')
#process.GlobalTag.DumpStat = cms.untracked.bool(True)  # optional if you want it to be verbose

# import of standard configurations
process.load("DQMServices.Core.DQMStore_cfg")
process.DQMStore.verbose   = cms.untracked.int32(1)
process.DQMStore.verboseQT = cms.untracked.int32(0)

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

# Input source
process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string('runnumber'),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(1),
                            interval = cms.uint64(1)
                            )

# from CondCore.CondDB.CondDB_cfi import *
# process.XmlRetrieval_1 = cms.ESSource("PoolDBESSource",
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

# process.ReferenceRetrieval = cms.ESSource("PoolDBESSource",
#                                   CondDBSetup,
#                                   connect = cms.string('sqlite_file:DQMReferenceHistogramTest.db'),
#                                   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
#                                   messageLevel = cms.untracked.int32(1), #3 for high verbosity
#                                   timetype = cms.string('runnumber'),
#                                   toGet = cms.VPSet(cms.PSet(record = cms.string('DQMReferenceHistogramRootFileRcd'),
#                                                              tag = cms.string('ROOTFILE_DQM')
#                                                              )
#                                                     )
#                                   )
#

# process.RecordDataGetter = cms.EDAnalyzer("EventSetupRecordDataGetter",
#                                   toGet = cms.VPSet(cms.PSet(record = cms.string('DQMReferenceHistogramRootFileRcd'),
#                                                          data = cms.vstring('ROOTFILE_DQM_Test10')
#                                                          )
#                                                 ),
#                                   verbose = cms.untracked.bool(False)
#                                   )
                                      
process.load('CondTools.DQM.DQMXMLFileEventSetupAnalyzer_cfi')
process.dqmXMLFileGetter.labelToGet = cms.string('SiPixelDQMQTests')
process.path = cms.Path(process.dqmXMLFileGetter)

