import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

iterativeCone5CaloJets = cms.EDProducer(
    "FastjetJetProducer",
    CaloJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("IterativeCone"),
    rParam       = cms.double(0.5)
    )

