import FWCore.ParameterSet.Config as cms

l1tPhase2CaloJetEmulator = cms.EDProducer("Phase2L1CaloJetEmulator",
						gctFullTowers = cms.InputTag("l1tPhase2L1CaloEGammaEmulator","GCTFullTowers"),
						hgcalTowers = cms.InputTag("l1tHGCalTowerProducer","HGCalTowerProcessor"),
                                                hcalDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
)
