import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

iterativeCone5HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                                         GenJetParameters,
                                         AnomalousCellParameters,
                                         jetAlgorithm = cms.string("IterativeCone"),
                                         srcMap = cms.InputTag("hiGenParticles"),
                                         rParam = cms.double(0.5)
                                         )

iterativeCone5HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

hiGenJets = cms.Sequence(hiGenParticlesForJets*iterativeCone5HiGenJets)
