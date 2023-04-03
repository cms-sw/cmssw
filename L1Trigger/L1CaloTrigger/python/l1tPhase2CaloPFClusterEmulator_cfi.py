import FWCore.ParameterSet.Config as cms

l1tPhase2CaloPFClusterEmulator = cms.EDProducer("Phase2L1CaloPFClusterEmulator",
						gctFullTowers = cms.InputTag("l1tPhase2L1CaloEGammaEmulator","GCTFullTowers"),
)
