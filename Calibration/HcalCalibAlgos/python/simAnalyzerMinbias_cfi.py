import FWCore.ParameterSet.Config as cms

simAnalyzerMinbias = cms.EDAnalyzer("SimAnalyzerMinbias",
                                    HistOutFile = cms.untracked.string("simOutput.root"),
                                    TimeCut     = cms.untracked.double(500),
                                    )
