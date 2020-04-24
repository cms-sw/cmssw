import FWCore.ParameterSet.Config as cms

overlapproblemtpanalyzer = cms.EDAnalyzer("OverlapProblemTPAnalyzer",
                                          trackingParticlesCollection = cms.InputTag("mergedtruth","MergedTrackTruth"),
                                          trackCollection = cms.InputTag("generalTracks")
                                          )

