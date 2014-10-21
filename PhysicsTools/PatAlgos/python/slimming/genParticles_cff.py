import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.slimming.prunedGenParticles_cfi import *
from PhysicsTools.PatAlgos.slimming.packedGenParticles_cfi import *

prunedGenParticlesWithStatusOne = prunedGenParticles.clone()
prunedGenParticlesWithStatusOne.select.append( "keep    status == 1")

prunedGenParticles.src =  cms.InputTag("prunedGenParticlesWithStatusOne")
packedGenParticles.inputCollection = cms.InputTag("prunedGenParticlesWithStatusOne")
packedGenParticles.map = cms.InputTag("prunedGenParticles")
