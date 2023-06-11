import FWCore.ParameterSet.Config as cms

BeamProfile2DBRead = cms.EDAnalyzer("BeamProfile2DBReader",
                                    rawFileName = cms.untracked.string("")
                                    )
