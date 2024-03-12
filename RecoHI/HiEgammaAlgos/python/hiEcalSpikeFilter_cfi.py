import FWCore.ParameterSet.Config as cms

hiEcalSpikeFilter = cms.EDFilter("HiEcalSpikeFilter",
                                 photonProducer = cms.InputTag("photons"),
                                 ebReducedRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                 eeReducedRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE")
                                 
                                 )




# foo bar baz
# ojzFGA8MF525y
# AwJH4HUXBAeMT
