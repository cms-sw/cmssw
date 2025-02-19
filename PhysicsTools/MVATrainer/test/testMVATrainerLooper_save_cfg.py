import FWCore.ParameterSet.Config as cms

process = cms.Process("testMVATrainerLooper")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("EmptySource", firstRun = cms.untracked.uint32(1) )
process.maxEvents = cms.untracked.PSet(	input = cms.untracked.int32(10000) )

process.testMVATrainerLooper = cms.EDAnalyzer("testMVATrainerLooper")

process.looper = cms.Looper("MVADemoTrainerLooper",
	trainers = cms.VPSet(cms.PSet(
		monitoring = cms.untracked.bool(True),
		calibrationRecord = cms.string('testMVA'),
		saveState = cms.untracked.bool(False),
		trainDescription = cms.untracked.string('testSave.xml'),
		loadState = cms.untracked.bool(False)
	))
)

process.p = cms.Path(process.testMVATrainerLooper)
