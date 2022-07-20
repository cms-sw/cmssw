import FWCore.ParameterSet.Config as cms

process = cms.Process("Calibration")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

process.load("CalibMuon.DTCalibration.dt_offlineAnalysis_common_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag=autoCond['run3_data']

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("CalibMuon.DTCalibration.dtTTrigCalibration_cfi")
process.dtTTrigCalibration.rootFileName = 'DTTimeBoxes.root'
process.dtTTrigCalibration.digiLabel = 'muonDTDigis'

#process.p = cms.Path(process.muonDTDigis*process.dtTTrigCalibration)
process.p = cms.Path(process.dtTTrigCalibration)
