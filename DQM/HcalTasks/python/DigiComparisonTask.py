import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
digiComparisonTask = DQMEDAnalyzer(
	"DigiComparisonTask",

	name = cms.untracked.string("DigiComparisonTask"),
	debug = cms.untracked.int32(0),
	runkeyVal = cms.untracked.int32(0),
	runkeyName = cms.untracked.string("pp_run"),

	tagHBHE1 = cms.untracked.InputTag("hcalDigis"),
	tagHBHE2 = cms.untracked.InputTag("uHBHEDigis")
)
