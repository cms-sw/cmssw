import FWCore.ParameterSet.Config as cms

JetMETPostVal = cms.EDAnalyzer("JetMETDQMPostProcessor",
     subDir = cms.untracked.string("HLT/HLTJETMET"),
     PatternJetTrg = cms.untracked.string("Jet([0-9])+"),
     PatternMetTrg = cms.untracked.string("M([E,H])+T([0-9])+")
       )
