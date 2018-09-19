import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
zdcTask = DQMEDAnalyzer(
	"ZDCTask",

	#	standard
	name = cms.untracked.string("ZDCTask"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),

	#	tag
	tagQIE10 = cms.untracked.InputTag("castorDigis"),

	#	cuts, 
	cut = cms.untracked.double(20),
	ped = cms.untracked.int32(4)
)

zdcQIE10Task = DQMEDAnalyzer(
	"ZDCQIE10Task",
	tagQIE10 = cms.untracked.InputTag('hcalDigis', 'ZDC')
)
