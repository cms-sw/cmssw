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

iterativeCone7HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                                         GenJetParameters,
                                         AnomalousCellParameters,
                                         jetAlgorithm = cms.string("IterativeCone"),
                                         srcMap = cms.InputTag("hiGenParticles"),
                                         rParam = cms.double(0.7)
                                         )

iterativeCone7HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

ak5HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                                         GenJetParameters,
                                         AnomalousCellParameters,
                                         jetAlgorithm = cms.string("IterativeCone"),
                                         srcMap = cms.InputTag("hiGenParticles"),
                                         rParam = cms.double(0.5)
                                         )

ak5HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

ak7HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                              GenJetParameters,
                              AnomalousCellParameters,
                              jetAlgorithm = cms.string("IterativeCone"),
                              srcMap = cms.InputTag("hiGenParticles"),
                              rParam = cms.double(0.7)
                              )

ak7HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

hiRecoGenJets = cms.Sequence(iterativeCone5HiGenJets + iterativeCone7HiGenJets + ak5HiGenJets + ak7HiGenJets)
