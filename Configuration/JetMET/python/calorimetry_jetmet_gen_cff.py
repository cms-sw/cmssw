import FWCore.ParameterSet.Config as cms

from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *
from RecoMET.Configuration.GenMETParticles_cff import *
from RecoMET.Configuration.RecoGenMET_cff import *
caloJetMetGen = cms.Sequence(genParticles*genJetParticles*recoGenJets)

