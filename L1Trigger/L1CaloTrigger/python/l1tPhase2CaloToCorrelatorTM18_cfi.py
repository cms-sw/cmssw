import FWCore.ParameterSet.Config as cms

l1tPhase2CaloToCorrelatorTM18  = cms.EDProducer("Phase2L1CaloToCorrelatorTM18",
                                  gctEmDigiClusters = cms.InputTag("l1tPhase2GCTBarrelToCorrelatorLayer1Emulator", "GCTEmDigiClusters"),
                                  gctHadDigiClusters = cms.InputTag("l1tPhase2GCTBarrelToCorrelatorLayer1Emulator", "GCTHadDigiClusters"),
)
