import FWCore.ParameterSet.Config as cms

l1TrkEGTausAnalysis     = cms.EDAnalyzer( 'L1TkEGTausAnalyzer' ,
                                          L1TkEGTauInputTag      = cms.InputTag("L1TkEGTaus", "TkEG"),
                                          GenParticleInputTag    = cms.InputTag("genParticles",""),
                                          AnalysisOption         = cms.string("Efficiency"),
                                          ObjectType             = cms.string("TkEGTau"),
                                          GenEtaCutOff           = cms.double(1.4),
                                          EtaCutOff              = cms.double(1.5),                            
                                          GenPtThreshold         = cms.double(0.0),
                                          EtThreshold            = cms.double(25.0)                              
                                          )
