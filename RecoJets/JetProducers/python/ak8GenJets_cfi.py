import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4GenJets_cfi import ak4GenJets

ak8GenJets = ak4GenJets.clone(
    rParam       = cms.double(0.8)
    )

ak8GenJetsSoftDrop = ak8GenJets.clone(
    useSoftDrop = cms.bool(True),
    zcut = cms.double(0.1),
    beta = cms.double(0.0),
    R0   = cms.double(0.8),
    useExplicitGhosts = cms.bool(True),
    writeCompound = cms.bool(True),
    jetCollInstanceName=cms.string("SubJets"),
    jetPtMin = 100.0
    )
