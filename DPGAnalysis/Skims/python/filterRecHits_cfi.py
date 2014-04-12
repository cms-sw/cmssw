import FWCore.ParameterSet.Config as cms

recHitEnergyFilter = cms.EDFilter("RecHitEnergyFilter",
   DoEB = cms.bool(True),
   DoEE = cms.bool(False),
   EBThresh = cms.double(3.0),
   EEThresh = cms.double(3.0),
   EBRecHits = cms.InputTag('ecalRecHit', 'EcalRecHitsEB'),
   EERecHits = cms.InputTag('ecalRecHit', 'EcalRecHitsEE')
)
