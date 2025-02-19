import FWCore.ParameterSet.Config as cms

process = cms.Process("testMVAComputerEvaluate")

process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.MVADemoFileSource = cms.ESSource("MVADemoFileSource",
	testMVA = cms.FileInPath('PhysicsTools/MVATrainer/test/testMVAComputerEvaluate.mva')
)

process.testMVAComputerEvaluate = cms.EDAnalyzer("testMVAComputerEvaluate")

process.p = cms.Path(process.testMVAComputerEvaluate)
