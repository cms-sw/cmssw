import FWCore.ParameterSet.Config as cms

ALCARECODTCalibrationDQM = DQMStep1Module('DTPreCalibrationTask',
    SaveFile = cms.untracked.bool(False),
    outputFileName = cms.untracked.string('DigiHistos.root'),
    digiLabel = cms.untracked.string('muonDTDigis'),
    minTriggerWidth = cms.untracked.int32(0),
    maxTriggerWidth = cms.untracked.int32(1600),
    folderName = cms.untracked.string('AlCaReco/DTCalibration')
)
