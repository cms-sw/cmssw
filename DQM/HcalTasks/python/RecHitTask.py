import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
recHitTask = DQMEDAnalyzer(
	"RecHitTask",
	
	#	standard parameters
	name = cms.untracked.string("RecHitTask"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),
	ptype = cms.untracked.int32(0),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string("Hcal"),

	#	tags
	tagHBHE = cms.untracked.InputTag("hbhereco"),
	tagHO = cms.untracked.InputTag("horeco"),
	tagHF = cms.untracked.InputTag("hfreco"),
	tagRaw = cms.untracked.InputTag('rawDataCollector'),

	#	thresholds
	thresh_unihf = cms.untracked.double(0.2),

	# prerechits
	hfPreRecHitsAvailable = cms.untracked.bool(False),
	tagPreHF = cms.untracked.InputTag(""),
)


recHitPreRecoTask = recHitTask.clone(
    tagHBHE = "hbheprereco",
)









