import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *


iterativeCone5HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                                         GenJetParameters,
                                         AnomalousCellParameters,
                                         jetAlgorithm = cms.string("IterativeCone"),
                                         rParam = cms.double(0.5)
                                         )

iterativeCone5HiGenJets.doAreaFastjet = True
iterativeCone5HiGenJets.doRhoFastjet  = False

iterativeCone7HiGenJets = iterativeCone5HiGenJets.clone(rParam=0.7)

ak5HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                              GenJetParameters,
                              AnomalousCellParameters,
                              jetAlgorithm = cms.string("AntiKt"),
                              rParam = cms.double(0.5)
                              )

ak5HiGenJets.doAreaFastjet = True
ak5HiGenJets.doRhoFastjet  = False

ak1HiGenJets = ak5HiGenJets.clone(rParam = 0.1)
ak2HiGenJets = ak5HiGenJets.clone(rParam = 0.2)
ak3HiGenJets = ak5HiGenJets.clone(rParam = 0.3)
ak4HiGenJets = ak5HiGenJets.clone(rParam = 0.4)
ak6HiGenJets = ak5HiGenJets.clone(rParam = 0.6)
ak7HiGenJets = ak5HiGenJets.clone(rParam = 0.7)

kt4HiGenJets = cms.EDProducer("SubEventGenJetProducer",
                              GenJetParameters,
                              AnomalousCellParameters,
                              jetAlgorithm = cms.string("Kt"),
                              rParam = cms.double(0.4)
                              )

kt4HiGenJets.doAreaFastjet = True
kt4HiGenJets.doRhoFastjet  = False

kt6HiGenJets = kt4HiGenJets.clone(rParam=0.6)

hiRecoGenJetsTask = cms.Task(
    iterativeCone5HiGenJets ,
    kt4HiGenJets ,
    kt6HiGenJets ,
    ak1HiGenJets ,
    ak2HiGenJets ,
    ak3HiGenJets ,
    ak4HiGenJets ,
    ak5HiGenJets ,
    ak6HiGenJets ,
    ak7HiGenJets
    )
hiRecoGenJets = cms.Sequence(hiRecoGenJetsTask)
