import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
pedestalTask = DQMEDAnalyzer(
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
	tagRaw = cms.untracked.InputTag('hltHcalCalibrationRaw'),

	thresh_mean = cms.untracked.double(0.25),
	thresh_rms = cms.untracked.double(0.25),
	thresh_badm = cms.untracked.double(0.1),
	thresh_badr = cms.untracked.double(0.1)
)


