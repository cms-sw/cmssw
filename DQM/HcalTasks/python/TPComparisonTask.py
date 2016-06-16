import FWCore.ParameterSet.Config as cms

tpComparisonTask = cms.EDAnalyzer(
	"TPComparisonTask",

	name = cms.untracked.string("TPComparisonTask"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),

	tag1 = cms.untracked.InputTag("hcalDigis"),
	tag2 = cms.untracked.InputTag("uHBHEDigis"),

	#	tmp
	_skip1x1 = cms.untracked.bool(True)
)
