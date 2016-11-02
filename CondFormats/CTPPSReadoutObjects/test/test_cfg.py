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

# load mapping(s)
process.load("CondFormats.CTPPSReadoutObjects.TotemDAQMappingESSourceXML_cfi")
process.totemDAQMappingESSourceXML.configuration = cms.VPSet(
  cms.PSet(
    validityRange = cms.EventRange("1:min - 999999999:max"),
    mappingFileNames = cms.vstring(
      "CondFormats/CTPPSReadoutObjects/xml/mapping_tracking_strip_from_fill_5330.xml",
      "CondFormats/CTPPSReadoutObjects/xml/mapping_timing_diamond.xml"
    ),
    maskFileNames = cms.vstring()
  )
)

# print the mapping
process.printTotemDAQMapping = cms.EDAnalyzer("PrintTotemDAQMapping"
)

process.path = cms.Path(
  process.printTotemDAQMapping
)
