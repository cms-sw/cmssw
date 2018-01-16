import FWCore.ParameterSet.Config as cms

    
ecalPreshowerPedestalTask = DQMStep1Module('ESPedestalTask',
	LookupTable = cms.untracked.FileInPath("EventFilter/ESDigiToRaw/data/ES_lookup_table.dat"),
	DigiLabel = cms.InputTag("ecalPreshowerDigis"),
	OutputFile = cms.untracked.string("")
)

