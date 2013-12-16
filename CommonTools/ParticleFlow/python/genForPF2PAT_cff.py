import FWCore.ParameterSet.Config as cms

# To reconstruct genjets without the neutrinos
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *


genForPF2PATSequence = cms.Sequence(
    genParticlesForJetsNoNu +
    ak4GenJetsNoNu +
    ak4GenJetsNoNu +
    ak8GenJetsNoNu
    )
