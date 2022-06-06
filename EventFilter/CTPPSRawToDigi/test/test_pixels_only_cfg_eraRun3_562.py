import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

#process = cms.Process("CTPPSRawToDigiTestPixelsOnly",eras.Run2_2017)
process = cms.Process("CTPPSRawToDigiTestPixelsOnly",eras.Run3)
# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG') )
)

# raw data source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
#    'file:/afs/cern.ch/user/f/fabferro/WORKSPACE/public/Unpacking/CMSSW_11_3_0/src/IORawData/SiPixelInputSources/test/PixelAlive_1463_548.root'
      'file:/eos/cms/store/group/dpg_ctpps/comm_ctpps/PixelAlive_562_RAW.root'
  ),
labelRawDataLikeMC = cms.untracked.bool(False), # for testing H8 data
duplicateCheckMode = cms.untracked.string("checkEachFile")
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

#process.load("CalibPPS.ESProducers.CTPPSPixelDAQMappingESSourceXML_cfi")

#ctppsPixelDAQMappingESSourceXML = cms.ESSource("CTPPSPixelDAQMappingESSourceXML",
#                                               verbosity = cms.untracked.uint32(2),
#                                               subSystem= cms.untracked.string("RPix"),
#                                               configuration = cms.VPSet(
        # example configuration block:
#        cms.PSet(
#            validityRange = cms.EventRange("1:min - 999999999:max"),
#            mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/rpix_tests_2021.xml"),
#            maskFileNames = cms.vstring("CondFormats/PPSObjects/xml/rpix_channel_mask_220_far.xml")
#            )
#        )
#                      
#)

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")
process.ctppsPixelDigis.inputLabel = cms.InputTag("source")

 

                       

#process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
#from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')


process.load("CondCore.CondDB.CondDB_cfi")
# input database (in this case the local sqlite file)
process.CondDB.connect = 'sqlite_file:/eos/cms/store/group/dpg_ctpps/comm_ctpps/CTPPSPixel_DAQMapping_AnalysisMask.db'

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    process.CondDB,
    DumpStat=cms.untracked.bool(True),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CTPPSPixelDAQMappingRcd'),
        tag = cms.string("PixelDAQMapping")
    )),
)

process.p = cms.Path(
  process.ctppsPixelDigis
#    process.ctppsRawToDigi
)

# output configuration
process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("file:./_digi_PixelAlive_562.root"),
  outputCommands = cms.untracked.vstring(
    'keep *'
 )
)

process.outpath = cms.EndPath(process.output)
