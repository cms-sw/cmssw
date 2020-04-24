import FWCore.ParameterSet.Config as cms

laserTask = cms.EDAnalyzer(
	"LaserTask",
	
	#	standard parameters
	name = cms.untracked.string("LaserTask"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),
	ptype = cms.untracked.int32(0),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string("HcalCalib"),

	#	tags
	tagHBHE = cms.untracked.InputTag("hcalDigis"),
	tagHO = cms.untracked.InputTag("hcalDigis"),
	tagHF = cms.untracked.InputTag("hcalDigis"),
    taguMN = cms.untracked.InputTag("hcalDigis"),
	tagRaw = cms.untracked.InputTag('hltHcalCalibrationRaw'),
	laserType = cms.untracked.uint32(0),

	nevents = cms.untracked.int32(10000)
)
