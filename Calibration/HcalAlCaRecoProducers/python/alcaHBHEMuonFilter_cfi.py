import FWCore.ParameterSet.Config as cms

AlcaHBHEMuonFilter = cms.EDFilter("AlCaHBHEMuonFilter",
                                  ProcessName       = cms.string("HLT"),
                                  TriggerResultLabel= cms.InputTag("TriggerResults","","HLT"),
                                  MuonLabel         = cms.InputTag("muons"),
                                  MinimumMuonP      = cms.double(10.0),
                                  Triggers           = cms.vstring("HLT_IsoMu","HLT_Mu"),
                                  )
