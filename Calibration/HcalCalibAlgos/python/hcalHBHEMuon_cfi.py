import FWCore.ParameterSet.Config as cms

HcalHBHEMuonAnalyzer = cms.EDAnalyzer("HcalHBHEMuonAnalyzer",
                                      HLTriggerResults = cms.InputTag("TriggerResults","","HLT"),
                                      LabelBS          = cms.InputTag("offlineBeamSpot"),
                                      LabelVertex      = cms.InputTag("offlinePrimaryVertices"),
                                      LabelEBRecHit    = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
                                      LabelEERecHit    = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
                                      LabelHBHERecHit  = cms.InputTag("hbhereco"),
                                      LabelMuon        = cms.InputTag("muons"),
                                      Verbosity        = cms.untracked.int32(0),
                                      MaxDepth         = cms.untracked.int32(4),
                                      )
