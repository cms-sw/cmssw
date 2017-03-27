import FWCore.ParameterSet.Config as cms

# trigger data
from EventFilter.CTPPSRawToDigi.totemTriggerRawToDigi_cfi import totemTriggerRawToDigi
totemTriggerRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")


# Si strips
totemDAQMappingESSourceXML_TrackingStrip = cms.ESSource("TotemDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),
  subSystem = cms.untracked.string("TrackingStrip"),
  configuration = cms.VPSet(
    # before TS2 (2016)
    cms.PSet(
      validityRange = cms.EventRange("1:min - 280385:max"),
      mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/mapping_tracking_strip_to_fill_5288.xml"),
      maskFileNames = cms.vstring()
    ),
    # during TS2 (2016)
    cms.PSet(
      validityRange = cms.EventRange("280386:min - 281600:max"),
      mappingFileNames = cms.vstring(),
      maskFileNames = cms.vstring()
    ),
    # after TS2 (2016)
    cms.PSet(
      validityRange = cms.EventRange("281601:min - 999999999:max"),
      mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/mapping_tracking_strip_from_fill_5330.xml"),
      maskFileNames = cms.vstring()
    )
  )
)

from EventFilter.CTPPSRawToDigi.totemRPRawToDigi_cfi import totemRPRawToDigi
totemRPRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")


# diamonds
totemDAQMappingESSourceXML_TimingDiamond = cms.ESSource("TotemDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),
  subSystem = cms.untracked.string("TimingDiamond"),
  configuration = cms.VPSet(
    # before diamonds inserted in DAQ
    cms.PSet(
      validityRange = cms.EventRange("1:min - 283819:max"),
      mappingFileNames = cms.vstring(),
      maskFileNames = cms.vstring()
    ),
    # after diamonds inserted in DAQ
    cms.PSet(
      validityRange = cms.EventRange("283820:min - 999999999:max"),
      mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/mapping_timing_diamond.xml"),
      maskFileNames = cms.vstring()
    )
  )
)

from EventFilter.CTPPSRawToDigi.ctppsDiamondRawToDigi_cfi import ctppsDiamondRawToDigi
ctppsDiamondRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")


# pixels
ctppsPixelDAQMappingESSourceXML = cms.ESSource("CTPPSPixelDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),
  subSystem= cms.untracked.string("RPix"),
  configuration = cms.VPSet(
  # example configuration block:
  cms.PSet(
    validityRange = cms.EventRange("1:min - 999999999:max"),
    mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/rpix_mapping_220_far.xml"),
    maskFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/rpix_channel_mask_220_far.xml")
    )
  )
)

from EventFilter.CTPPSRawToDigi.ctppsPixelRawToDigi_cfi import ctppsPixelDigis
ctppsPixelDigis.rawDataTag = cms.InputTag("rawDataCollector")

# raw-to-digi sequence
ctppsRawToDigi = cms.Sequence(
  totemTriggerRawToDigi *
  totemRPRawToDigi *
  ctppsDiamondRawToDigi*
  ctppsPixelDigis
)
