import FWCore.ParameterSet.Config as cms

# To reconstruct genjets without the neutrinos
from RecoJets.Configuration.GenJetParticles_cff import genParticlesForJetsNoNu
from RecoJets.Configuration.RecoGenJets_cff import iterativeCone5GenJetsNoNu, ak4GenJetsNoNu, ak8GenJetsNoNu


genForPF2PATSequence = cms.Sequence(
    genParticlesForJetsNoNu +
    iterativeCone5GenJetsNoNu +
    ak4GenJetsNoNu +
    ak8GenJetsNoNu
    )
