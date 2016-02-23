import FWCore.ParameterSet.Config as cms

from RecoHI.HiJetAlgos.HiGenJets_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *

ak1HiGenJets.signalOnly = cms.bool(False)
ak2HiGenJets.signalOnly = cms.bool(False)
ak3HiGenJets.signalOnly = cms.bool(False)
ak4HiGenJets.signalOnly = cms.bool(False)
ak5HiGenJets.signalOnly = cms.bool(False)
ak6HiGenJets.signalOnly = cms.bool(False)

akHiGenJets = cms.Sequence(
    genParticlesForJets +
    ak1HiGenJets +
    ak2HiGenJets +
    ak3HiGenJets +
    ak4HiGenJets +
    ak5HiGenJets +
    ak6HiGenJets)
