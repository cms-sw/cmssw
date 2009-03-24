import FWCore.ParameterSet.Config as cms

siPixelDigis = cms.EDFilter("SiPixelRawToDigi",
    Timing = cms.untracked.bool(False),
    IncludeErrors = cms.bool(False),
    InputLabel = cms.InputTag("siPixelRawData"),
    CheckPixelOrder = cms.bool(False),
    UseCablingTree = cms.untracked.bool(True)
)



