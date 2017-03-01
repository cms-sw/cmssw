import FWCore.ParameterSet.Config as cms

AlcaHBHEMuonFilter = cms.EDFilter("AlCaHBHEMuonFilter",
                                  ProcessName       = cms.string("HLT"),
                                  TriggerResultLabel= cms.InputTag("TriggerResults","","HLT"),
                                  MuonLabel         = cms.InputTag("muons"),
                                  MinimumMuonP      = cms.double(10.0),
                                  Triggers          = cms.vstring("HLT_IsoMu","HLT_Mu"),
                                  PFCut             = cms.bool(True),
                                  PFIsolationCut    = cms.double(0.12),
                                  TrackIsolationCut = cms.double(3.0),
                                  CaloIsolationCut  = cms.double(5.0),
                                  PreScale          = cms.int32(2),
                                  )
