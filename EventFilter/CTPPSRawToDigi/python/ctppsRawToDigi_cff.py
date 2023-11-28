import FWCore.ParameterSet.Config as cms

# ---------- Si strips ----------
from EventFilter.CTPPSRawToDigi.totemRPRawToDigi_cfi import totemRPRawToDigi
totemRPRawToDigi.rawDataTag = cms.InputTag("rawDataCollector")

# various error/warning/info output may be enabled with these flags
#  totemRPRawToDigi.RawUnpacking.verbosity = 1
#  totemRPRawToDigi.RawToDigi.verbosity = 1 # or higher number for more output
#  totemRPRawToDigi.RawToDigi.printErrorSummary = True
#  totemRPRawToDigi.RawToDigi.printUnknownFrameSummary = True

# ---------- diamonds ----------
from EventFilter.CTPPSRawToDigi.ctppsDiamondRawToDigi_cfi import ctppsDiamondRawToDigi
ctppsDiamondRawToDigi.rawDataTag = "rawDataCollector"

# ---------- Totem Timing ----------
from EventFilter.CTPPSRawToDigi.totemTimingRawToDigi_cfi import totemTimingRawToDigi
totemTimingRawToDigi.rawDataTag = "rawDataCollector"

# ---------- Totem nT2 ----------
from EventFilter.CTPPSRawToDigi.totemT2Digis_cfi import totemT2Digis
totemT2Digis.rawDataTag = "rawDataCollector"

# ---------- pixels ----------
from EventFilter.CTPPSRawToDigi.ctppsPixelDigis_cfi import ctppsPixelDigis
ctppsPixelDigis.inputLabel = "rawDataCollector"

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
from Configuration.Eras.Modifier_ctpps_2017_cff import ctpps_2017
from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
(ctpps_2016 | ctpps_2017 | ctpps_2018).toModify(ctppsPixelDigis, isRun3 = False )

# raw-to-digi task and sequence
ctppsRawToDigiTask = cms.Task(
  totemRPRawToDigi,
  ctppsDiamondRawToDigi,
  totemTimingRawToDigi,
  totemT2Digis,
  ctppsPixelDigis
)
ctppsRawToDigi = cms.Sequence(ctppsRawToDigiTask)
