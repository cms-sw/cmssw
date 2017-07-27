import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.slimming.genParticles_cff import *

patGENTask = cms.Task(
    genParticlesTask
)

miniGEN=cms.Sequence()
