import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFCaloJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

ak4PFCaloJets = cms.EDProducer(
    "FastjetJetProducer",
    PFCaloJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.4)
    )

