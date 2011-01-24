import FWCore.ParameterSet.Config as cms

process = cms.Process("Calibration")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = ''

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring()
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("CalibMuon.DTCalibration.dtTTrigCalibration_cfi")
process.ttrigcalib.rootFileName = 'DTTimeBoxes.root'
process.ttrigcalib.digiLabel = 'muonDTDigis'
#process.ttrigcalib.kFactor = -0.7

# if read from RAW
#process.load("EventFilter.DTRawToDigi.dtunpacker_cfi")

#process.p = cms.Path(process.muonDTDigis*process.dtTTrigCalibration)
process.p = cms.Path(process.dtTTrigCalibration)
