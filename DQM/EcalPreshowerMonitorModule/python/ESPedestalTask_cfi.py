import FWCore.ParameterSet.Config as cms

    
ecalPreshowerPedestalTask = cms.EDAnalyzer('ESPedestalTask',
	LookupTable = cms.untracked.FileInPath("EventFilter/ESDigiToRaw/data/ES_lookup_table.dat"),
	DigiLabel = cms.InputTag("simEcalPreshowerDigis"),
	OutputFile = cms.untracked.string("")
)

