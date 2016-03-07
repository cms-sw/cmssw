import FWCore.ParameterSet.Config as cms

from RecoHI.HiJetAlgos.HiGenJets_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *

akHiGenJets = cms.Sequence(
    genParticlesForJets +
    ak1HiGenJets +
    ak2HiGenJets +
    ak3HiGenJets +
    ak4HiGenJets +
    ak5HiGenJets +
    ak6HiGenJets)
