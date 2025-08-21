import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
hfRaddamTask = DQMEDAnalyzer(
	"HFRaddamTask",
	
	#	standard parameters
	name = cms.untracked.string("HFRaddamTask"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),
	ptype = cms.untracked.int32(0),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string('HcalCalib'),

    laserType = cms.untracked.uint32(0),
	nevents = cms.untracked.int32(2000),

	#	tags
	tagHF = cms.untracked.InputTag("hcalDigis"),
	tagRaw = cms.untracked.InputTag('hltHcalCalibrationRaw'),
    tagFEDs = cms.untracked.InputTag("hltHcalCalibrationRaw")
)












