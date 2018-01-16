import FWCore.ParameterSet.Config as cms

    
ecalPreshowerOccupancyTask = DQMStep1Module('ESOccupancyTask',
	RecHitLabel = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
	DigiLabel = cms.InputTag("ecalPreshowerDigis"),
	prefixME = cms.untracked.string("EcalPreshower") 
)

