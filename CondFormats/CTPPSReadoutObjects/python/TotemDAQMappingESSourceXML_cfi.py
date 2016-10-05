import FWCore.ParameterSet.Config as cms

totemDAQMappingESSourceXML = cms.ESSource("TotemDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),

  configuration = cms.VPSet(
    # example configuration block:
    #cms.PSet(
    #  validityRange = cms.EventRange("1:min - 999999999:max"),
    #  mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/ctpps_mapping_to_fill_5288.xml"),
    #  maskFileNames = cms.vstring()
    #)
  )
)
