import FWCore.ParameterSet.Config as cms

SusyExoPostVal = cms.EDAnalyzer("HltSusyExoPostProcessor",
   subDir = cms.untracked.string("HLT/SusyExo")                   
                            )
