import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.slimming.genParticles_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import ak4GenJetsNoNu, ak8GenJetsNoNu
from PhysicsTools.PatAlgos.slimming.slimmedGenJets_cfi   import *

patGENTask = cms.Task(
    genParticlesTask,
    genParticlesForJets,
    genParticlesForJetsNoNu,
    ak4GenJetsNoNu,
    slimmedGenJets,
    ak8GenJetsNoNu,
    slimmedGenJetsAK8
)

miniGEN=cms.Sequence()
