import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFClusterJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

ak4PFClusterJets = cms.EDProducer(
    "FastjetJetProducer",
    PFClusterJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("AntiKt"),
    rParam       = cms.double(0.4)
    )

# foo bar baz
# 6I4Q7Huz2xWL7
# vWUch35Ge00Gm
