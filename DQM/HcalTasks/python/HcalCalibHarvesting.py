import FWCore.ParameterSet.Config as cms

hcalCalibHarvesting = cms.EDAnalyzer(
	"HcalCalibHarvesting",

	name = cms.untracked.string("HcalCalibHarvesting"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string('pp_run'),
	ptype = cms.untracked.int32(0),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string("HcalCalib"),
)
