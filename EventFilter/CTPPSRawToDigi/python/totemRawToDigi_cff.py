import FWCore.ParameterSet.Config as cms

from CondFormats.CTPPSReadoutObjects.TotemDAQMappingESSourceXML_cfi import *
totemDAQMappingESSourceXML.configuration = cms.VPSet(
  cms.PSet(
    validityRange = cms.EventRange("1:min - 280385:max"),
    mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/ctpps_mapping_to_fill_5288.xml"),
    maskFileNames = cms.vstring()
  ),
  cms.PSet(
    validityRange = cms.EventRange("281601:min - 999999999:max"),
    mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/ctpps_mapping_from_fill_5330.xml"),
    maskFileNames = cms.vstring()
  )
)

from EventFilter.CTPPSRawToDigi.totemTriggerRawToDigi_cfi import *
totemTriggerRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

from EventFilter.CTPPSRawToDigi.totemRPRawToDigi_cfi import *
totemRPRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")
