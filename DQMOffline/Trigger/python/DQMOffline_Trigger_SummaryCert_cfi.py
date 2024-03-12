import FWCore.ParameterSet.Config as cms

hltOverallSummary = cms.EDAnalyzer("HLTOverallSummary",
  verbose = cms.untracked.bool(False)
)

hltOverallCertSeq = cms.Sequence(hltOverallSummary)
# foo bar baz
# SeBllSitKIr0m
# SnpFCYmKVm8CX
