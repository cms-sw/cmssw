import FWCore.ParameterSet.Config as cms

    
ecalPreshowerOccupancyTask = cms.EDAnalyzer('ESOccupancyTask',
	RecHitLabel = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
	DigiLabel = cms.InputTag("simEcalPreshowerDigis"),
	prefixME = cms.untracked.string("EcalPreshower") 
)

