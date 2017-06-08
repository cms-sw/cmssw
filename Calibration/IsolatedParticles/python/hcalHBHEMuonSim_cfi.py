import FWCore.ParameterSet.Config as cms

HcalHBHEMuonSimAnalyzer = cms.EDAnalyzer("HcalHBHEMuonSimAnalyzer",
                                         ModuleLabel    = cms.string('g4SimHits'),
                                         EBCollection   = cms.string('EcalHitsEB'),
                                         EECollection   = cms.string('EcalHitsEE'),
                                         HCCollection   = cms.string('HcalHits'),
                                         Verbosity      = cms.untracked.int32(0),
                                         MaxDepth       = cms.untracked.int32(4),
                                         EtaMax         = cms.untracked.double(3.0),
                                         TimeMinCutECAL = cms.untracked.double(-500.0),
                                         TimeMaxCutECAL = cms.untracked.double(500.0),
                                         TimeMinCutHCAL = cms.untracked.double(-500.0),
                                         TimeMaxCutHCAL = cms.untracked.double(500.0),
                                         )
