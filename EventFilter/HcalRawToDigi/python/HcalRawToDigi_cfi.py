import FWCore.ParameterSet.Config as cms

# This version is intended for unpacking standard production data
hcalDigis = cms.EDProducer("HcalRawToDigi",
    # Flag to enable unpacking of ZDC channels (default = false)
    UnpackZDC = cms.untracked.bool(True),
    # Flag to enable unpacking of TTP channels (default = false)
    UnpackTTP = cms.untracked.bool(True),
    # Optional filter to remove any digi with "data valid" off, "error" on, 
    # or capids not rotating
    FilterDataQuality = cms.bool(True),
    InputLabel = cms.InputTag("rawDataCollector"),
    # Use the defaults for FED numbers
    # Do not complain about missing FEDs
    ComplainEmptyData = cms.untracked.bool(False),
    # Flag to enable unpacking of calibration channels (default = false)
    UnpackCalib = cms.untracked.bool(True),
    lastSample = cms.int32(9),
    # At most ten samples can be put into a digi, if there are more
    # than ten, firstSample and lastSample select which samples
    # will be copied to the digi
    firstSample = cms.int32(0)
)


