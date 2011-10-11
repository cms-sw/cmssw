import FWCore.ParameterSet.Config as cms

JetMETPostVal = cms.EDAnalyzer("JetMETDQMPostProcessor",
     subDir = cms.untracked.string("HLT/HLTJETMET")
       )
