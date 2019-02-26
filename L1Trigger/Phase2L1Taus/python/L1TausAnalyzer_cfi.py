import FWCore.ParameterSet.Config as cms

l1TausAnalysis     = cms.EDAnalyzer( 'L1TausAnalyzer' ,
                                     L1TrkTauInputTag       = cms.InputTag("L1TrkTaus", "TrkTau"),
                                     L1TkEGTauInputTag      = cms.InputTag("L1TkEGTaus", "TkEG"),
                                     L1CaloTkTauInputTag    = cms.InputTag("L1CaloTkTaus", "CaloTk"),
                                     GenParticleInputTag    = cms.InputTag("genParticles",""),
                                     AnalysisOption         = cms.string("Efficiency"),
                                     ObjectType             = cms.string("Electron"),
                                     GenEtaCutOff           = cms.double(1.4),
                                     EtaCutOff              = cms.double(1.5),                            
                                     GenPtThreshold         = cms.double(0.0),
                                     EtThreshold            = cms.double(25.0)                              
                                     )
