import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ic5GenJets_cfi import iterativeCone5GenJets
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

hiGenParticlesForJets = genParticlesForJets.clone()
hiGenParticlesForJets.src = cms.InputTag("hiGenParticles")

iterativeCone5HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                                         GenJetParameters,
                                         AnomalousCellParameters,
                                         jetAlgorithm = cms.string("IterativeCone"),
                                         srcMap = cms.InputTag("hiGenParticles"),
                                         rParam = cms.double(0.5)
                                         )

iterativeCone5HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

signalJets = cms.Sequence(genJetParticles*iterativeCone5GenJets)

subEventJets = cms.Sequence(hiGenParticlesForJets*iterativeCone5HiGenJets)

allGenJets = cms.Sequence(signalJets+subEventJets)
