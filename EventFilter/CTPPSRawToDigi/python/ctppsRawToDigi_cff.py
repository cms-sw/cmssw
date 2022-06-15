import FWCore.ParameterSet.Config as cms

# ---------- trigger data ----------
from EventFilter.CTPPSRawToDigi.totemTriggerRawToDigi_cfi import totemTriggerRawToDigi
totemTriggerRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

# ---------- Si strips ----------
totemDAQMappingESSourceXML_TrackingStrip = cms.ESSource("TotemDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),
  subSystem = cms.untracked.string("TrackingStrip"),
  sampicSubDetId = cms.uint32(6),
  configuration = cms.VPSet(
    # 2016, before TS2
    cms.PSet(
      validityRange = cms.EventRange("1:min - 280385:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2016_to_fill_5288.xml"),
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
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2016_from_fill_5330.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2017
    cms.PSet(
      validityRange = cms.EventRange("290873:min - 311625:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2017.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2018
    cms.PSet(
      validityRange = cms.EventRange("311626:min - 339999:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2018.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2022
    cms.PSet(
      validityRange = cms.EventRange("340000:min - 999999999:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_tracking_strip_2022.xml"),
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
  sampicSubDetId = cms.uint32(6),
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
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2017
    cms.PSet(
      validityRange = cms.EventRange("292521:min - 310000:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond_2017.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2018
    cms.PSet(
      validityRange = cms.EventRange("310001:min - 339999:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond_2018.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2022
    cms.PSet(
      validityRange = cms.EventRange("340000:min - 999999999:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_timing_diamond_2022.xml"),
      maskFileNames = cms.vstring()
    )
    
  )
)

from EventFilter.CTPPSRawToDigi.ctppsDiamondRawToDigi_cfi import ctppsDiamondRawToDigi
ctppsDiamondRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

# ---------- Totem Timing ----------
totemDAQMappingESSourceXML_TotemTiming = cms.ESSource("TotemDAQMappingESSourceXML",
  verbosity = cms.untracked.uint32(0),
  subSystem = cms.untracked.string("TotemTiming"),
  sampicSubDetId = cms.uint32(5),
  configuration = cms.VPSet(
    # 2017, before detector inserted in DAQ
    cms.PSet(
      validityRange = cms.EventRange("1:min - 310000:max"),
      mappingFileNames = cms.vstring(),
      maskFileNames = cms.vstring()
    ),
    # 2018
    cms.PSet(
      validityRange = cms.EventRange("310001:min - 339999:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_totem_timing_2018.xml"),
      maskFileNames = cms.vstring()
    ),
    # 2022
    cms.PSet(
      validityRange = cms.EventRange("340000:min - 999999999:max"),
      mappingFileNames = cms.vstring("CondFormats/PPSObjects/xml/mapping_totem_timing_2022.xml"),
      maskFileNames = cms.vstring()
    )
  )
)

from EventFilter.CTPPSRawToDigi.totemTimingRawToDigi_cfi import totemTimingRawToDigi
totemTimingRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

# ---------- pixels ----------
from EventFilter.CTPPSRawToDigi.ctppsPixelDigis_cfi import ctppsPixelDigis
ctppsPixelDigis.inputLabel = cms.InputTag("rawDataCollector")

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
(ctpps_2016 | ctpps_2017 | ctpps_2018).toModify(ctppsPixelDigis, isRun3 = False )
(ctpps_2016 | ctpps_2017 | ctpps_2018).toModify(totemDAQMappingESSourceXML_TotemTiming, sampicSubDetId = 6)

# raw-to-digi task and sequence
ctppsRawToDigiTask = cms.Task(
  totemTriggerRawToDigi,
  totemRPRawToDigi,
  ctppsDiamondRawToDigi,
  totemTimingRawToDigi,
  ctppsPixelDigis
)
ctppsRawToDigi = cms.Sequence(ctppsRawToDigiTask)
