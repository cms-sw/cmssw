import FWCore.ParameterSet.Config as cms

# Joanna Weng, PFlow kt 1.0 jets
from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt10PFJets = cms.EDProducer("KtJetProducer",
    FastjetNoPU,
    KtJetParameters,
    PFJetParameters,
    JetPtMin = cms.double(1.0),
    alias = cms.untracked.string('KT10PFJet'),
    FJ_ktRParam = cms.double(1.0)
)


