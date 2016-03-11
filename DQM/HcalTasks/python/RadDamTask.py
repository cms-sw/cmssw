import FWCore.ParameterSet.Config as cms

raddamTask = cms.EDAnalyzer(
	"RadDamTask",
	
	#	standard parameters
	name = cms.untracked.string("RadDamTask"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),
	ptype = cms.untracked.int32(0),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string('HcalCalib'),

	#	tags
	tagHF = cms.untracked.InputTag("hcalDigis"),
	tagRaw = cms.untracked.InputTag('hltHcalCalibrationRaw')
)












