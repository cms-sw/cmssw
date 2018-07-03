import FWCore.ParameterSet.Config as cms

simAnalyzerMinbias = cms.EDAnalyzer("SimAnalyzerMinbias",
                                    TimeCut     = cms.untracked.double(500),
                                    )
