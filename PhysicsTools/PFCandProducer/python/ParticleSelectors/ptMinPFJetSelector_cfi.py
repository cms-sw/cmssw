import FWCore.ParameterSet.Config as cms


ptMinPFJets = cms.EDFilter(
    "PtMinPFJetSelector",
    src = cms.InputTag(''),
    ptMin = cms.double(0)
    )
