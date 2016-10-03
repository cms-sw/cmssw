import FWCore.ParameterSet.Config as cms

totemDAQMappingESSourceXML = cms.ESSource("TotemDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),

  configuration = cms.untracked.VPSet(
    # example configuration block:
    #cms.PSet(
    #  validityRange = cms.untracked.EventRange("1:1:1 - 999999999:999999999:999999999999"),
    #  mappingFileNames = cms.untracked.vstring("CondFormats/CTPPSReadoutObjects/xml/ctpps_mapping_to_fill_5288.xml"),
    #  maskFileNames = cms.untracked.vstring()
    #)
  )
)
