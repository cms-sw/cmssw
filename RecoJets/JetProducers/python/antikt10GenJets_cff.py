import FWCore.ParameterSet.Config as cms

# $Id: antikt10GenJets_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.AntiKtJetParameters_cfi import *
antikt10GenJets = cms.EDProducer("AntiKtJetProducer",
    GenJetParameters,
    FastjetNoPU,
    AntiKtJetParameters,
    
    alias = cms.untracked.string('ANTIKT10GenJet'),
    FJ_ktRParam = cms.double(1.0)
)


