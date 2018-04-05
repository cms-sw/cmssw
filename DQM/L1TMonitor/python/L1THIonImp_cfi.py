import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
l1tHIonImp = DQMEDAnalyzer('L1THIonImp',
                            gctCentralJetsDataSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets"),
                            gctForwardJetsDataSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets"),
                            gctTauJetsDataSource = cms.InputTag("caloStage1LegacyFormatDigis","tauJets"),
                            gctEnergySumsDataSource = cms.InputTag("caloStage1LegacyFormatDigis"),
                            gctIsoEmDataSource = cms.InputTag("caloStage1LegacyFormatDigis","isoEm"),
                            gctNonIsoEmDataSource = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm"),
                            rctSource = cms.InputTag("gctDigis"),
                            gctCentralJetsEmulSource = cms.InputTag("valCaloStage1LegacyFormatDigis","cenJets"),
                            gctForwardJetsEmulSource = cms.InputTag("valCaloStage1LegacyFormatDigis","forJets"),
                            gctTauJetsEmulSource = cms.InputTag("valCaloStage1LegacyFormatDigis","tauJets"),
                            gctEnergySumsEmulSource = cms.InputTag("valCaloStage1LegacyFormatDigis"),
                            gctIsoEmEmulSource = cms.InputTag("valCaloStage1LegacyFormatDigis","isoEm"),
                            gctNonIsoEmEmulSource = cms.InputTag("valCaloStage1LegacyFormatDigis","nonIsoEm")
)

                    
