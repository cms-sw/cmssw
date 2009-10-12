import FWCore.ParameterSet.Config as cms

ALCARECODTCalibrationDQM = cms.EDFilter("DTPreCalibrationTask",
    SaveFile = cms.untracked.bool(False),
    outputFileName = cms.untracked.string('DigiHistos.root'),
    digiLabel = cms.untracked.string('muonDTDigis'),
    minTriggerWidth = cms.untracked.int32(2000),
    maxTriggerWidth = cms.untracked.int32(6000),
    folderName = cms.untracked.string('AlCaReco')
)
