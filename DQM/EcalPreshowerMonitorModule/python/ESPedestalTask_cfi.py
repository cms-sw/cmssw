import FWCore.ParameterSet.Config as cms

    
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalPreshowerPedestalTask = DQMEDAnalyzer('ESPedestalTask',
	LookupTable = cms.untracked.FileInPath("EventFilter/ESDigiToRaw/data/ES_lookup_table.dat"),
	DigiLabel = cms.InputTag("ecalPreshowerDigis"),
	OutputFile = cms.untracked.string("")
)

