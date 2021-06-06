from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

process = cms.Process('XMLFILERETRIEVER')

####################################################################
# Set the options
####################################################################
options = VarParsing('analysis')
options.register('unitTest',
                 False,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool,
                 'Run the unit test',
                 )

options.parseArguments()

####################################################################
# Get the GlobalTag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run3_data_prompt', '')
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

if options.unitTest:
    print("running configuration in UnitTest Mode")
    process.load("CondCore.CondDB.CondDB_cfi")
    process.CondDB.connect = "sqlite_file:./testXML.db"

    process.XmlRetrieval_1 = cms.ESSource("PoolDBESSource",
                                          process.CondDB,
                                          BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                          messageLevel = cms.untracked.int32(1),
                                          timetype = cms.string('runnumber'),
                                          toGet = cms.VPSet(cms.PSet(record = cms.string('DQMXMLFileRcd'),
                                                                     tag = cms.string('XML_test'),
                                                                     label=cms.untracked.string('XML_label')
                                                                 )
                                                        )
                                      )

    process.RecordDataGetter = cms.EDAnalyzer("EventSetupRecordDataGetter",
                                              toGet = cms.VPSet(cms.PSet(record = cms.string('DQMXMLFileRcd'),
                                                                         data = cms.vstring('FileBlob/XML_label'))),
                                              verbose = cms.untracked.bool(True)
                                              )


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
if(options.unitTest):
    process.dqmXMLFileGetter.labelToGet = cms.string('XML_label')
    process.path = cms.Path(process.RecordDataGetter+process.dqmXMLFileGetter)

else:
    process.dqmXMLFileGetter.labelToGet = cms.string('SiPixelDQMQTests')
    process.path = cms.Path(process.dqmXMLFileGetter)

