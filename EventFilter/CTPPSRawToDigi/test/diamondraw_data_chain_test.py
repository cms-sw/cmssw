import FWCore.ParameterSet.Config as cms

process = cms.Process("DiamondRawToDigiTest")
process.maxEvents = cms.untracked.PSet(
input = cms.untracked.int32(500)
)
# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG') )
)

# raw data source
process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring('/store/t0streamer/Minidaq/A/000/281/709/run281709_ls0016_streamA_StorageManager.dat')
)   
 
# raw-to-digi conversion
process.load('CondFormats.CTPPSReadoutObjects.TotemDAQMappingESSourceXML_cfi')
#process.DiamondDAQMappingESSourceXML.mappingFileNames.append("CondFormats/CTPPSReadoutObjects/xml/ctpps_timing_diamond_215_mapping.xml")
process.TotemDAQMappingESSourceXML.mappingFileNames.append("CondFormats/CTPPSReadoutObjects/xml/ctpps_timing_diamond_215_mapping_new.xml")

#process.load("EventFilter.TotemRawToDigi.totemTriggerRawToDigi_cfi")
#process.totemTriggerRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

process.load('EventFilter.CTPPSRawToDigi.ctppsDiamondRawToDigi_cfi')
process.ctppsDiamondRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

# ntuplizer
#process.load("TotemAnalysis.TotemNtuplizer.TotemNtuplizer_cfi")
#process.totemNtuplizer.outputFileName = "ntuple.root"

process.p = cms.Path(
#    process.totemTriggerRawToDigi
     process.ctppsDiamondRawToDigi
#    * process.dump
)


# output configuration
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("file:./DiamondDigi.root"),
     outputCommands = cms.untracked.vstring(
    'keep TotemFEDInfos_ctppsDiamondRawToDigi_*_*',
    'keep CTPPSDiamondDigiedmDetSetVector_ctppsDiamondRawToDigi_*_*',
    'keep TotemVFATStatusedmDetSetVector_ctppsDiamondRawToDigi_*_*'
 )
)

process.outpath = cms.EndPath(process.output)
