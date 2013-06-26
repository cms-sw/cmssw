import FWCore.ParameterSet.Config as cms

ProbeTreeProducer = cms.EDAnalyzer("ProbeTreeProducer",
    src = cms.InputTag("patMuons"),
    cut = cms.string("isGlobalMuon"),
    variables = cms.PSet(
        pt  = cms.string("pt"),
        eta = cms.string("eta"),
    ),
    flags = cms.PSet(
        GlobalMuonPromptTight = cms.string("muonID('GlobalMuonPromptTight')"),
    ),
)

