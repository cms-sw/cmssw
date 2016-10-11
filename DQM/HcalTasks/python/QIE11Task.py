import FWCore.ParameterSet.Config as cms

qie11Task = cms.EDAnalyzer(
	"QIE11Task",

	#	standard
	name = cms.untracked.string("QIE11Task"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),

	#	tag
	tagQIE11 = cms.untracked.InputTag("hcalDigis"),

	#	cuts, 
	cut = cms.untracked.double(20),
	ped = cms.untracked.int32(4),

        laserType = cms.untracked.int32(-1)
)
