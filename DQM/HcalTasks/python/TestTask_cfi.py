import FWCore.ParameterSet.Config as cms

testTask = cms.EDAnalyzer(
	"TestTask",

	#	standard
	name = cms.untracked.string("TestTask"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),

	tagHF = cms.untracked.InputTag("qie10Digis")
)
