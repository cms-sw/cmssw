import FWCore.ParameterSet.Config as cms

CentralityDQM = cms.EDAnalyzer(
    "CentralityDQM",
    centrality_ = cms.InputTag("hiCentrality")
)
