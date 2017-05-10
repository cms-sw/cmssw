import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.GenJetParameters_cfi import *

ak5GenJets = cms.EDProducer( "FastjetJetProducer",
    GenJetParameters,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.5)
    )
