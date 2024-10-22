import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
nocqTask = DQMEDAnalyzer(
	"NoCQTask",
	
	#	standard parameters
	name = cms.untracked.string("NoCQTask"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),
	ptype = cms.untracked.int32(0),
	mtype = cms.untracked.bool(True),
	subsystem = cms.untracked.string("Hcal"),

	#	tags
	tagHBHE = cms.untracked.InputTag("hcalDigis"),
	tagHO = cms.untracked.InputTag("hcalDigis"),
	tagHF = cms.untracked.InputTag("hcalDigis"),
	tagReport = cms.untracked.InputTag("hcalDigis"),

	#	Cuts
	cutSumQ_HBHE = cms.untracked.double(20),
	cutSumQ_HO = cms.untracked.double(20),
	cutSumQ_HF = cms.untracked.double(20),

)












