import FWCore.ParameterSet.Config as cms

BeamSpotRead = cms.EDAnalyzer("BeamSpotRcdReader",
                              rawFileName = cms.untracked.string("")
                              )
