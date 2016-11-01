import FWCore.ParameterSet.Config as cms

from CondFormats.CTPPSReadoutObjects.TotemDAQMappingESSourceXML_cfi import *
TotemDAQMappingESSourceXML.mappingFileNames.append("CondFormats/CTPPSReadoutObjects/xml/ctpps_210_mapping.xml")
TotemDAQMappingESSourceXML.mappingFileNames.append("CondFormats/CTPPSReadoutObjects/xml/ctpps_timing_diamond_215_mapping.xml")

# raw trigger data
from EventFilter.CTPPSRawToDigi.totemTriggerRawToDigi_cfi import *
totemTriggerRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

# Si strips
from EventFilter.CTPPSRawToDigi.totemRPRawToDigi_cfi import *
totemRPRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

# diamonds
from EventFilter.CTPPSRawToDigi.ctppsDiamondRawToDigi_cfi import *
ctppsDiamondRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

# raw-to-digi sequence
ctppsRawToDigi = cms.Sequence(
  totemTriggerRawToDigi *
  totemRPRawToDigi *
  ctppsDiamondRawToDigi
)
