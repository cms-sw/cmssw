import FWCore.ParameterSet.Config as cms

l1TausAnalysis = cms.EDAnalyzer( 'L1TausAnalyzer' ,
                                 L1TrkTauInputTag       = cms.InputTag("L1TrkTaus", "TrkTau"),
                                 L1TkEGTauInputTag      = cms.InputTag("L1TkEGTaus", "TkEG"),
                                 L1CaloTkTauInputTag    = cms.InputTag("L1CaloTkTaus", "CaloTk"),
                                 GenParticleInputTag    = cms.InputTag("genParticles",""),
                                 AnalysisOption         = cms.string("Efficiency"),
                                 ObjectType             = cms.string("Tau"), # fixme - new
                                 GenEtaVisCutOff        = cms.double(1.4), # fixme. use VIS
                                 GenEtVisCutOff         = cms.double(1.4), # fixme-implement. use VIS
                                 L1EtCutOff             = cms.double(0.0), # fixme - L1Et
                                 L1EtaCutOff            = cms.double(1.5), # fixme - L1Eta
                                 #GenPtThreshold         = cms.double(0.0), #remove
                                 EtThreshold            = cms.double(25.0), #fixme L1TurnOnThreshold
                                 DRMatching             = cms.double(0.1)  #fixme - implement in Analyzer
                                     )
