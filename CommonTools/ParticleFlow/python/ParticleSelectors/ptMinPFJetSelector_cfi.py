import FWCore.ParameterSet.Config as cms


ptMinPFJets = cms.EDFilter(
    "PtMinPFJetSelector",
    src = cms.InputTag(''),
    ptMin = cms.double(0)
    )
# foo bar baz
# 3XL1Wqn2F7dfp
# DgMpLW1k1f0ep
