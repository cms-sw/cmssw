import FWCore.ParameterSet.Config as cms

hcalDigis = cms.EDFilter("HcalRawToDigi",
    FilterDataQuality = cms.bool(True),
    lastSample = cms.int32(9),
    InputLabel = cms.InputTag("source"),
    ComplainEmptyData = cms.untracked.bool(False),
    UnpackCalib = cms.untracked.bool(False),
    ExceptionEmptyData = cms.untracked.bool(False),
    firstSample = cms.int32(0)
)


