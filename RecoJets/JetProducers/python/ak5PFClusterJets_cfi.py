import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFClusterJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

ak5PFClusterJets = cms.EDProducer(
    "FastjetJetProducer",
    PFClusterJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.5)
    )

