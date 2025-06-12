import FWCore.ParameterSet.Config as cms
from  PhysicsTools.NanoAOD.particlelevel_cff import *

particleLevelBPHSequence = cms.Sequence(mergedGenParticles + genParticles2HepMC + particleLevel)
