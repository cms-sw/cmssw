import FWCore.ParameterSet.Config as cms

spikeInspector = cms.EDAnalyzer('SpikeInspector',
                                ebSuperClusterCollection  = cms.untracked.InputTag("correctedIslandBarrelSuperClusters"),
                                ebReducedRecHitCollection = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEB"),
                                eeReducedRecHitCollection = cms.untracked.InputTag("ecalRecHit","EcalRecHitsEE"),
                                swissCut                  = cms.untracked.double(0.95),
                                photonCollection          = cms.untracked.InputTag("photons")
                                )
