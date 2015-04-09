import FWCore.ParameterSet.Config as cms

import L1TriggerDPG.L1Ntuples.l1NtupleProducer_cfi
l1NtupleProducer = L1TriggerDPG.L1Ntuples.l1NtupleProducer_cfi.l1NtupleProducer.clone()

l1NtupleProducer.gctCentralJetsSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets")
l1NtupleProducer.gctNonIsoEmSource    = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm")
l1NtupleProducer.gctForwardJetsSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets")
l1NtupleProducer.gctIsoEmSource       = cms.InputTag("caloStage1LegacyFormatDigis","isoEm")
l1NtupleProducer.gctEnergySumsSource  = cms.InputTag("caloStage1LegacyFormatDigis","")
l1NtupleProducer.gctTauJetsSource     = cms.InputTag("caloStage1LegacyFormatDigis","tauJets")
l1NtupleProducer.rctSource            = cms.InputTag("caloStage1LegacyFormatDigis")

