import FWCore.ParameterSet.Config as cms

# Joanna Weng, PFlow antikt 1.0 jets
from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.AntiKtJetParameters_cfi import *
antikt10PFJets = cms.EDProducer("AntiKtJetProducer",
    FastjetNoPU,
    AntiKtJetParameters,
    PFJetParameters,
    
    alias = cms.untracked.string('ANTIKT10PFJet'),
    FJ_ktRParam = cms.double(1.0)
)


