import FWCore.ParameterSet.Config as cms

# To reconstruct genjets without the neutrinos
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *


genForPF2PATSequence = cms.Sequence(
    genParticlesForJetsNoNu +
    iterativeCone5GenJetsNoNu +
    ak5GenJetsNoNu +
    ak8GenJetsNoNu
    )
