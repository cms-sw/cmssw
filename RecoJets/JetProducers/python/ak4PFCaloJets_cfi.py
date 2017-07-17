import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *


PFCaloJetParameters = PFJetParameters.clone(
    src = cms.InputTag('hltParticleFlow')
)

ak4PFCaloJets = cms.EDProducer(
    "FastjetJetProducer",
    PFCaloJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.4)
    )

