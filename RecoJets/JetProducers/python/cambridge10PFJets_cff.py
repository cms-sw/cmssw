import FWCore.ParameterSet.Config as cms

# Joanna Weng, PFlow cambridge 1.0 jets
from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.CambridgeJetParameters_cfi import *
cambridge10PFJets = cms.EDProducer("CambridgeJetProducer",
    FastjetNoPU,
    CambridgeJetParameters,
    PFJetParameters,
    
    alias = cms.untracked.string('CAMBRIDGE10PFJet'),
    FJ_ktRParam = cms.double(1.0)
)


