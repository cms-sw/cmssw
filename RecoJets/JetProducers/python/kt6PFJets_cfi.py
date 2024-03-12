import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

kt6PFJets = cms.EDProducer(
    "FastjetJetProducer",
    PFJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("Kt"),
    rParam       = cms.double(0.6)
    )

# foo bar baz
# F77pdqtIyxFp6
# mUT0sCqvkvUso
