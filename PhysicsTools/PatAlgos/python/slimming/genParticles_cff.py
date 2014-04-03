import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.slimming.prunedGenParticles_cfi import *
from PhysicsTools.PatAlgos.slimming.packedGenCandidates_cfi import *

prunedGenParticlesWithStatusOne = prunedGenParticles.clone()
prunedGenParticlesWithStatusOne.select.append( "keep    status == 1")

prunedGenParticles.src =  cms.InputTag("prunedGenParticlesWithStatusOne")
packedGenCandidates.inputCollection = cms.InputTag("prunedGenParticlesWithStatusOne")
packedGenCandidates.map = cms.InputTag("prunedGenParticles")
