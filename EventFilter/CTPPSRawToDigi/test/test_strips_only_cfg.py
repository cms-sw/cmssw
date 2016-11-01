import FWCore.ParameterSet.Config as cms

process = cms.Process("CTPPSRawToDigiTestStripsOnly")

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cerr'),
  cerr = cms.untracked.PSet(
    threshold = cms.untracked.string('WARNING')
  )
)

# raw data source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    #'file:/afs/cern.ch/user/j/jkaspar/public/run273062_ls0001-2_stream.root'
    '/store/express/Run2016H/ExpressPhysics/FEVT/Express-v2/000/283/877/00000/4EE44B0E-2499-E611-A155-02163E011938.root'
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

# raw-to-digi conversion
process.load('CondFormats.CTPPSReadoutObjects.TotemDAQMappingESSourceXML_cfi')
process.TotemDAQMappingESSourceXML.mappingFileNames.append("CondFormats/CTPPSReadoutObjects/xml/ctpps_210_mapping.xml")

process.load("EventFilter.CTPPSRawToDigi.totemTriggerRawToDigi_cfi")
process.totemTriggerRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

process.load('EventFilter.CTPPSRawToDigi.totemRPRawToDigi_cfi')
process.totemRPRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

# execution configuration
process.p = cms.Path(
  process.totemTriggerRawToDigi *
  process.totemRPRawToDigi
)
