import FWCore.ParameterSet.Config as cms

# trigger data
from EventFilter.CTPPSRawToDigi.totemTriggerRawToDigi_cfi import *
totemTriggerRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

# Si strips
totemDAQMappingESSourceXML_TrackingStrip = cms.ESSource("TotemDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),
  subSystem = cms.untracked.string("TrackingStrip"),
  mappingFileNames = cms.untracked.vstring("CondFormats/CTPPSReadoutObjects/xml/ctpps_210_mapping.xml"),
  maskFileNames = cms.untracked.vstring()
)

from EventFilter.CTPPSRawToDigi.totemRPRawToDigi_cfi import *
totemRPRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

# diamonds
totemDAQMappingESSourceXML_TimingDiamond = cms.ESSource("TotemDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),
  subSystem = cms.untracked.string("TimingDiamond"),
  mappingFileNames = cms.untracked.vstring("CondFormats/CTPPSReadoutObjects/xml/ctpps_timing_diamond_215_mapping.xml"),
  maskFileNames = cms.untracked.vstring()
)

from EventFilter.CTPPSRawToDigi.ctppsDiamondRawToDigi_cfi import *
ctppsDiamondRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

# raw-to-digi sequence
ctppsRawToDigi = cms.Sequence(
  totemTriggerRawToDigi *
  totemRPRawToDigi *
  ctppsDiamondRawToDigi
)
