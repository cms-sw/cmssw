import FWCore.ParameterSet.Config as cms
process = cms.Process("MERGE")

process.source = cms.Source("EmptySource")

process.doit = cms.EDAnalyzer ("IntTestAnalyzer",
		valueMustMatch = cms.untracked.int32(90),
		moduleLabel = cms.untracked.string("missing"))

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(3))

process.o = cms.Path(process.doit)

process.options = cms.untracked.PSet( 
	StopProcessing = cms.untracked.vstring('ProductNotFound') )
 
