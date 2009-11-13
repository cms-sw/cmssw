import FWCore.ParameterSet.Config as cms



hltOverallSummary = cms.EDAnalyzer("DQMOfflineHLTEventInfoClient",
									verbose = cms.untracked.bool(False)
									)

hltOverallCertSeq = cms.Sequence(hltOverallSummary)
