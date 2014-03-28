import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.slimming.prunedGenParticles_cfi import *
prunedGenParticlesWithStatusOne = prunedGenParticles.clone()
prunedGenParticlesWithStatusOne.select.append( "drop    status == 2")
packedGenCandidates = cms.EDProducer("PATPackedGenCandidateProducer",
    inputCollection = cms.InputTag("genParticles"),
    inputVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
    maxEta = cms.double(5)
)
