import FWCore.ParameterSet.Config as cms

from CondFormats.CTPPSReadoutObjects.TotemDAQMappingESSourceXML_cfi import *
totemDAQMappingESSourceXML.configuration = cms.untracked.VPSet(
  cms.PSet(
    validityRange = cms.untracked.EventRange("1:1:1 - 280385:999999999:999999999999"),
    mappingFileNames = cms.untracked.vstring("CondFormats/CTPPSReadoutObjects/xml/ctpps_mapping_to_fill_5288.xml"),
    maskFileNames = cms.untracked.vstring()
  ),
  cms.PSet(
    validityRange = cms.untracked.EventRange("281601:1:1 - 999999999:999999999:999999999999"),
    mappingFileNames = cms.untracked.vstring("CondFormats/CTPPSReadoutObjects/xml/ctpps_mapping_from_fill_5330.xml"),
    maskFileNames = cms.untracked.vstring()
  )
)

from EventFilter.CTPPSRawToDigi.totemTriggerRawToDigi_cfi import *
totemTriggerRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

from EventFilter.CTPPSRawToDigi.totemRPRawToDigi_cfi import *
totemRPRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")
