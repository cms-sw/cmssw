import FWCore.ParameterSet.Config as cms

ecalPreshowerTimingTask = DQMStep1Module('ESTimingTask',
	RecHitLabel = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
	DigiLabel = cms.InputTag("ecalPreshowerDigis"),
	prefixME = cms.untracked.string("EcalPreshower") 
)

