import FWCore.ParameterSet.Config as cms

AlcaHBHEMuonFilter = cms.EDFilter("AlCaHBHEMuonFilter",
                                  ProcessName       = cms.string("HLT"),
                                  TriggerResultLabel= cms.InputTag("TriggerResults","","HLT"),
                                  MuonLabel         = cms.InputTag("muons"),
                                  MinimumMuonP      = cms.double(10.0),
                                  Triggers           = cms.vstring("HLT_IsoMu17","HLT_IsoMu20","HLT_IsoMu24","HLT_IsoMu27","HLT_Mu45","HLT_Mu50"),
                                  )
