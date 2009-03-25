import FWCore.ParameterSet.Config as cms

dtPreCalibTask = cms.EDFilter("DTPreCalibrationTask",
    SaveFile = cms.untracked.bool(True),
    outputFileName = cms.untracked.string('DigiHistos.root'),
    digiLabel = cms.untracked.string('muonDTDigis'),
    minTriggerWidth = cms.untracked.int32(2000),
    maxTriggerWidth = cms.untracked.int32(6000)
)
