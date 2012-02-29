import FWCore.ParameterSet.Config as cms

siPixelDigis = cms.EDProducer("SiPixelRawToDigi",
    Timing = cms.untracked.bool(False),
    IncludeErrors = cms.bool(True),
    OverflowList =  cms.bool(False),
    InputLabel = cms.InputTag("siPixelRawData"),
    CheckPixelOrder = cms.bool(False),
    UseQualityInfo = cms.bool(False),
    UseCablingTree = cms.untracked.bool(True),
    ErrorList = cms.vint32(29)
)



