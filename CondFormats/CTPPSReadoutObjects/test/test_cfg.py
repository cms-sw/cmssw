import FWCore.ParameterSet.Config as cms

process = cms.Process('test')

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cerr'),
  cerr = cms.untracked.PSet(
    threshold = cms.untracked.string('WARNING')
  )
)

# raw data source
process.source = cms.Source("EmptySource",
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(1)
)

# load a mapping
process.load("CondFormats.CTPPSReadoutObjects.totemDAQMappingESSourceXML_cfi")
process.totemDAQMappingESSourceXML.subSystem = "TrackingStrip"
process.totemDAQMappingESSourceXML.configuration = cms.VPSet(
  cms.PSet(
    validityRange = cms.EventRange("1:min - 999999999:max"),
    mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/mapping_tracking_strip_2017.xml"),
    maskFileNames = cms.vstring()
  )
)
  
# print the mapping
process.printTotemDAQMapping = cms.EDAnalyzer("PrintTotemDAQMapping",
  subSystem = cms.untracked.string("TrackingStrip")
)

process.path = cms.Path(
  process.printTotemDAQMapping
)
