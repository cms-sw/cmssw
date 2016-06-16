import FWCore.ParameterSet.Config as cms

hcalOnlineHarvesting = cms.EDAnalyzer(
	"HcalOnlineHarvesting",

	name = cms.untracked.string("HcalOnlineHarvesting"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string('pp_run'),
	ptype = cms.untracked.int32(0),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string("Hcal"),
)
