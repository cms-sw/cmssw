import FWCore.ParameterSet.Config as cms

overlapproblemtpanalyzer = cms.EDAnalyzer("OverlapProblemTPAnalyzer",
                                          trackingParticlesCollection = cms.InputTag("mergedtruth","MergedTrackTruth"),
                                          trackCollection = cms.InputTag("generalTracks")
                                          )

# foo bar baz
# ri0RVOZSQJ9J4
# Ts8WdlK4X0VwP
