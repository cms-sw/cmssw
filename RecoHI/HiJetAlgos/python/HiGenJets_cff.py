import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

iterativeCone3HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                                         GenJetParameters,
                                         AnomalousCellParameters,
                                         jetAlgorithm = cms.string("IterativeCone"),
                                         rParam = cms.double(0.3)
                                         )

iterativeCone3HiGenJets.doAreaFastjet = cms.bool(True)
iterativeCone3HiGenJets.doRhoFastjet  = cms.bool(False)
iterativeCone3HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

iterativeCone4HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                                         GenJetParameters,
                                         AnomalousCellParameters,
                                         jetAlgorithm = cms.string("IterativeCone"),
                                         rParam = cms.double(0.4)
                                         )

iterativeCone4HiGenJets.doAreaFastjet = cms.bool(True)
iterativeCone4HiGenJets.doRhoFastjet  = cms.bool(False)
iterativeCone4HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

iterativeCone5HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                                         GenJetParameters,
                                         AnomalousCellParameters,
                                         jetAlgorithm = cms.string("IterativeCone"),
                                         rParam = cms.double(0.5)
                                         )

iterativeCone5HiGenJets.doAreaFastjet = cms.bool(True)
iterativeCone5HiGenJets.doRhoFastjet  = cms.bool(False)
iterativeCone5HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

iterativeCone7HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                                         GenJetParameters,
                                         AnomalousCellParameters,
                                         jetAlgorithm = cms.string("IterativeCone"),
                                         rParam = cms.double(0.7)
                                         )

iterativeCone7HiGenJets.src = cms.InputTag("hiGenParticlesForJets")
iterativeCone7HiGenJets.doAreaFastjet = cms.bool(True)
iterativeCone7HiGenJets.doRhoFastjet  = cms.bool(False)

ak3HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                              GenJetParameters,
                              AnomalousCellParameters,
                              jetAlgorithm = cms.string("AntiKt"),
                              rParam = cms.double(0.3)
                              )

ak3HiGenJets.doAreaFastjet = cms.bool(True)
ak3HiGenJets.doRhoFastjet  = cms.bool(False)
ak3HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

ak4HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                              GenJetParameters,
                              AnomalousCellParameters,
                              jetAlgorithm = cms.string("AntiKt"),
                              rParam = cms.double(0.4)
                              )

ak4HiGenJets.doAreaFastjet = cms.bool(True)
ak4HiGenJets.doRhoFastjet  = cms.bool(False)
ak4HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

ak5HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                              GenJetParameters,
                              AnomalousCellParameters,
                              jetAlgorithm = cms.string("AntiKt"),
                              rParam = cms.double(0.5)
                              )

ak5HiGenJets.doAreaFastjet = cms.bool(True)
ak5HiGenJets.doRhoFastjet  = cms.bool(False)
ak5HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

ak7HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                              GenJetParameters,
                              AnomalousCellParameters,
                              jetAlgorithm = cms.string("AntiKt"),
                              rParam = cms.double(0.7)
                              )

ak7HiGenJets.doAreaFastjet = cms.bool(True)
ak7HiGenJets.doRhoFastjet  = cms.bool(False)
ak7HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

kt3HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                              GenJetParameters,
                              AnomalousCellParameters,
                              jetAlgorithm = cms.string("Kt"),
                              rParam = cms.double(0.3)
                              )

kt3HiGenJets.doAreaFastjet = cms.bool(True)
kt3HiGenJets.doRhoFastjet  = cms.bool(False)
kt3HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

kt4HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                              GenJetParameters,
                              AnomalousCellParameters,
                              jetAlgorithm = cms.string("Kt"),
                              rParam = cms.double(0.4)
                              )

kt4HiGenJets.doAreaFastjet = cms.bool(True)
kt4HiGenJets.doRhoFastjet  = cms.bool(False)
kt4HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

kt6HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                              GenJetParameters,
                              AnomalousCellParameters,
                              jetAlgorithm = cms.string("Kt"),
                              rParam = cms.double(0.6)
                              )

kt6HiGenJets.doAreaFastjet = cms.bool(True)
kt6HiGenJets.doRhoFastjet  = cms.bool(False)
kt6HiGenJets.src = cms.InputTag("hiGenParticlesForJets")

hiRecoGenJets = cms.Sequence(iterativeCone5HiGenJets + iterativeCone7HiGenJets + ak5HiGenJets + ak7HiGenJets + kt4HiGenJets + kt6HiGenJets)
