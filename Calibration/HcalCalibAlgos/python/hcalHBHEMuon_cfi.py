import FWCore.ParameterSet.Config as cms

HcalHBHEMuonAnalyzer = cms.EDAnalyzer("HcalHBHEMuonAnalyzer",
                                      HLTriggerResults = cms.InputTag("TriggerResults","","HLT"),
                                      LabelBS          = cms.string("offlineBeamSpot"),
                                      LabelVertex      = cms.string("offlinePrimaryVertices"),
                                      LabelEBRecHit    = cms.string("EcalRecHitsEB"),
                                      LabelEERecHit    = cms.string("EcalRecHitsEE"),
                                      LabelHBHERecHit  = cms.string("hbhereco"),
                                      LabelMuon        = cms.string("muons"),
                                      ModuleName       = cms.untracked.string("HBHEMuonProd"),
                                      ProcessName      = cms.untracked.string("AlCaHBHEMuon"),
                                      Verbosity        = cms.untracked.int32(0),
                                      MaxDepth         = cms.untracked.int32(4),
                                      )
