import FWCore.ParameterSet.Config as cms

hcalOfflineHarvesting = cms.EDAnalyzer(
	"HcalOfflineHarvesting",

	name = cms.untracked.string("HcalOfflineHarvesting"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string('pp_run'),
	ptype = cms.untracked.int32(1),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string("Hcal"),

	thresh_fgmsm = cms.untracked.double(0.1),
	thresh_etmsm = cms.untracked.double(0.1),
	thresh_unihf = cms.untracked.double(0.2),
	thresh_tcds = cms.untracked.double(1.5)
)
