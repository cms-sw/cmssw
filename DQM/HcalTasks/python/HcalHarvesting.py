import FWCore.ParameterSet.Config as cms

hcalHarvesting = cms.EDAnalyzer(
	"HcalHarvesting",

	name = cms.untracked.string("HcalHarvesting"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string('pp_run'),
	ptype = cms.untracked.int32(0),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string("Hcal"),

	rawHarvesting = cms.untracked.bool(True),
	digiHarvesting = cms.untracked.bool(True),
	recoHarvesting = cms.untracked.bool(True),
	tpHarvesting = cms.untracked.bool(True)
)
