import FWCore.ParameterSet.Config as cms

process = cms.Process("CTPPSRawToDigiTestDiamondsOnly")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    statistics = cms.untracked.vstring(),
    destinations = cms.untracked.vstring('cerr'),
    cerr = cms.untracked.PSet( threshold = cms.untracked.string('DEBUG') )
)

# raw data source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    #'file:/afs/cern.ch/user/j/jkaspar/public/run273062_ls0001-2_stream.root'
    '/store/express/Run2016H/ExpressPhysics/FEVT/Express-v2/000/283/877/00000/4EE44B0E-2499-E611-A155-02163E011938.root'
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(100)
)
 
# raw-to-digi conversion
process.load('CondFormats.CTPPSReadoutObjects.TotemDAQMappingESSourceXML_cfi')
process.TotemDAQMappingESSourceXML.mappingFileNames.append("CondFormats/CTPPSReadoutObjects/xml/ctpps_timing_diamond_215_mapping.xml")

process.load('EventFilter.CTPPSRawToDigi.ctppsDiamondRawToDigi_cfi')
process.ctppsDiamondRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

process.p = cms.Path(
  process.ctppsDiamondRawToDigi
)

# output configuration
process.output = cms.OutputModule("PoolOutputModule",
  fileName = cms.untracked.string("file:./reco_diamond_digi.root"),
  outputCommands = cms.untracked.vstring(
    'keep TotemFEDInfos_ctppsDiamondRawToDigi_*_*',
    'keep CTPPSDiamondDigiedmDetSetVector_ctppsDiamondRawToDigi_*_*',
    'keep TotemVFATStatusedmDetSetVector_ctppsDiamondRawToDigi_*_*'
 )
)

process.outpath = cms.EndPath(process.output)
