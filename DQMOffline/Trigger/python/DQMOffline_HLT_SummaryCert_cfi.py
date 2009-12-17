import FWCore.ParameterSet.Config as cms



HLTOverallSummary = cms.EDAnalyzer("DQMOfflineHLTEventInfoClient",
									verbose = cms.untracked.bool(False)
									)

HLTOverallCertSeq = cms.Sequence(HLTOverallSummary)
