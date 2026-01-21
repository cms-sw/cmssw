import FWCore.ParameterSet.Config as cms

l1tPhase2CaloToCorrelatorTMI18  = cms.EDProducer("Phase2L1CaloToCorrelatorTMI18",
                                  gctEmDigiClusters = cms.InputTag("l1tPhase2GCTBarrelToCorrelatorLayer1Emulator", "GCTEmDigiClusters"),
                                  gctHadDigiClusters = cms.InputTag("l1tPhase2GCTBarrelToCorrelatorLayer1Emulator", "GCTHadDigiClusters"),
)
