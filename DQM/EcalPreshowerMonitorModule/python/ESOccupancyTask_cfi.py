import FWCore.ParameterSet.Config as cms

    
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
ecalPreshowerOccupancyTask = DQMEDAnalyzer('ESOccupancyTask',
	RecHitLabel = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
	DigiLabel = cms.InputTag("ecalPreshowerDigis"),
	prefixME = cms.untracked.string("EcalPreshower") 
)

