import FWCore.ParameterSet.Config as cms

qie10Task = cms.EDAnalyzer(
	"QIE10Task",

	#	standard
	name = cms.untracked.string("QIE10Task"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),

	#	tag
	tagQIE10 = cms.untracked.InputTag("hcalDigis"),

	#	cuts, 
	cut = cms.untracked.double(20),
	ped = cms.untracked.int32(4)
)
