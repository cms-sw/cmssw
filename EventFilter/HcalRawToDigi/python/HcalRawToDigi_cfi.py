import FWCore.ParameterSet.Config as cms
import EventFilter.HcalRawToDigi.hcalRawToDigi_cfi

# This version is intended for unpacking standard production data
hcalDigis =  EventFilter.HcalRawToDigi.hcalRawToDigi_cfi.hcalRawToDigi.clone()
# Flag to enable unpacking of ZDC channels (default = false)
hcalDigis.UnpackZDC = cms.untracked.bool(True)
# Flag to enable unpacking of TTP channels (default = false)
hcalDigis.UnpackTTP = cms.untracked.bool(True)
# Optional filter to remove any digi with "data valid" off, "error" on, 
# or capids not rotating
hcalDigis.FilterDataQuality = cms.bool(True)
hcalDigis.InputLabel = cms.InputTag("rawDataCollector")
# Use the defaults for FED numbers
# Do not complain about missing FEDs
hcalDigis.ComplainEmptyData = cms.untracked.bool(False)
# Flag to enable unpacking of calibration channels (default = false)
hcalDigis.UnpackCalib = cms.untracked.bool(True)
hcalDigis.lastSample = cms.int32(9)
# At most ten samples can be put into a digi, if there are more
# than ten, firstSample and lastSample select which samples
# will be copied to the digi
hcalDigis.firstSample = cms.int32(0)
