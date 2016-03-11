import FWCore.ParameterSet.Config as cms

pedestalTask = cms.EDAnalyzer(
	"PedestalTask",
	
	#	standard parameters
	name = cms.untracked.string("PedestalTask"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),
	ptype = cms.untracked.int32(0),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string('HcalCalib'),

	#	tags
	tagHBHE = cms.untracked.InputTag("hcalDigis"),
	tagHO = cms.untracked.InputTag("hcalDigis"),
	tagHF = cms.untracked.InputTag("hcalDigis"),
	tagRaw = cms.untracked.InputTag('hltHcalCalibrationRaw')
)












