import FWCore.ParameterSet.Config as cms

# $Id: antikt4CaloJets_cff.py,v 1.1 2008/11/10 19:05:03 srappocc Exp $
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.AntiKtJetParameters_cfi import *
antikt7CaloJets = cms.EDProducer("AntiKtJetProducer",
    FastjetNoPU,
    AntiKtJetParameters,
    CaloJetParameters,
    
    alias = cms.untracked.string('ANTIKT7CaloJet'),
    FJ_ktRParam = cms.double(0.7)
)


