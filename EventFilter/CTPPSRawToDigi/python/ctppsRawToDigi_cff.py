import FWCore.ParameterSet.Config as cms

# ---------- trigger data ----------
from EventFilter.CTPPSRawToDigi.totemTriggerRawToDigi_cfi import totemTriggerRawToDigi
totemTriggerRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")



# ---------- Si strips ----------
totemDAQMappingESSourceXML_TrackingStrip = cms.ESSource("TotemDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),
  subSystem = cms.untracked.string("TrackingStrip"),
  configuration = cms.VPSet(
    # 2016, before TS2
    cms.PSet(
      validityRange = cms.EventRange("1:min - 280385:max"),
      mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/mapping_tracking_strip_2016_to_fill_5288.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2016, during TS2
    cms.PSet(
      validityRange = cms.EventRange("280386:min - 281600:max"),
      mappingFileNames = cms.vstring(),
      maskFileNames = cms.vstring()
    ),
    # 2016, after TS2
    cms.PSet(
      validityRange = cms.EventRange("281601:min - 290872:max"),
      mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/mapping_tracking_strip_2016_from_fill_5330.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2017
    cms.PSet(
      validityRange = cms.EventRange("290873:min - 999999999:max"),
      mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/mapping_tracking_strip_2017.xml"),
      maskFileNames = cms.vstring()
    )
  )
)

from EventFilter.CTPPSRawToDigi.totemRPRawToDigi_cfi import totemRPRawToDigi
totemRPRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

# various error/warning/info output may be enabled with these flags
#  totemRPRawToDigi.RawUnpacking.verbosity = 1
#  totemRPRawToDigi.RawToDigi.verbosity = 1 # or higher number for more output
#  totemRPRawToDigi.RawToDigi.printErrorSummary = 1
#  totemRPRawToDigi.RawToDigi.printUnknownFrameSummary = 1



# ---------- diamonds ----------
totemDAQMappingESSourceXML_TimingDiamond = cms.ESSource("TotemDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),
  subSystem = cms.untracked.string("TimingDiamond"),
  configuration = cms.VPSet(
    # 2016, before diamonds inserted in DAQ
    cms.PSet(
      validityRange = cms.EventRange("1:min - 283819:max"),
      mappingFileNames = cms.vstring(),
      maskFileNames = cms.vstring()
    ),
    # 2016, after diamonds inserted in DAQ
    cms.PSet(
      validityRange = cms.EventRange("283820:min - 292520:max"),
      mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/mapping_timing_diamond.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2017
    cms.PSet(
      validityRange = cms.EventRange("292521:min - 999999999:max"),
      mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/mapping_timing_diamond_2017.xml"),
      maskFileNames = cms.vstring()
    )
  )
)

from EventFilter.CTPPSRawToDigi.ctppsDiamondRawToDigi_cfi import ctppsDiamondRawToDigi
ctppsDiamondRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")



# ---------- pixels ----------
## this part needed only for direct access to xmls. No DB use.
#ctppsPixelDAQMappingESSourceXML = cms.ESSource("CTPPSPixelDAQMappingESSourceXML",
#  verbosity = cms.untracked.uint32(0),
#  subSystem= cms.untracked.string("RPix"),
#  configuration = cms.VPSet(
#  # example configuration block:
#  cms.PSet(
#    validityRange = cms.EventRange("1:min - 999999999:max"),
#    mappingFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/rpix_mapping_220_far.xml"),
#    maskFileNames = cms.vstring("CondFormats/CTPPSReadoutObjects/xml/rpix_channel_mask_220_far.xml")
#    )
#  )
#)

from EventFilter.CTPPSRawToDigi.ctppsPixelRawToDigi_cfi import ctppsPixelDigis
ctppsPixelDigis.InputLabel = cms.InputTag("rawDataCollector")

# raw-to-digi sequence
ctppsRawToDigi = cms.Sequence(
  totemTriggerRawToDigi *
  totemRPRawToDigi *
  ctppsDiamondRawToDigi*
  ctppsPixelDigis
)
