import FWCore.ParameterSet.Config as cms

hcalDigis = cms.EDFilter("HcalRawToDigi",
    UnpackZDC = cms.untracked.bool(False),
    FilterDataQuality = cms.bool(True),
    ExceptionEmptyData = cms.untracked.bool(True),
    HcalFirstFED = cms.untracked.int32(700),
    InputLabel = cms.InputTag("source"),
    UnpackCalib = cms.untracked.bool(False),
    FEDs = cms.untracked.vint32(700),
    lastSample = cms.int32(9),
    firstSample = cms.int32(0)
)


