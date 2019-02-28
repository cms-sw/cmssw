import FWCore.ParameterSet.Config as cms

l1TausAnalysis = cms.EDAnalyzer( 'L1TausAnalyzer' ,
                                 L1TrkTauInputTag          = cms.InputTag("L1TrkTaus", "TrkTau"),
                                 L1TkEGTauInputTag         = cms.InputTag("L1TkEGTaus", "TkEG"),
                                 L1CaloTkTauInputTag       = cms.InputTag("L1CaloTkTaus", "CaloTk"),
                                 GenParticleInputTag       = cms.InputTag("genParticles",""),
                                 AnalysisOption            = cms.string("Efficiency"), # just a default value (re-defined in L1TausAnalyzer_cff.py)
                                 ObjectType                = cms.string("TrkTau"), # just a default value (re-defined in L1TausAnalyzer_cff.py)
                                 GenEtVisThreshold         = cms.double(0),
                                 GenEtaVisCutOff           = cms.double(1.4),
                                 GenEtVisThreshold_Trigger = cms.double(20.0),
                                 L1EtThreshold             = cms.double(0.0),
                                 L1EtaCutOff               = cms.double(1.5), 
                                 L1TurnOnThreshold         = cms.double(25.0), 
                                 DRMatching                = cms.double(0.1) 
                                 )
