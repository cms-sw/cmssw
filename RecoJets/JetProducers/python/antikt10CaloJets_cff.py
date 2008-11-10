import FWCore.ParameterSet.Config as cms

# $Id: antikt10CaloJets_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.AntiKtJetParameters_cfi import *
antikt10CaloJets = cms.EDProducer("AntiKtJetProducer",
    FastjetNoPU,
    AntiKtJetParameters,
    CaloJetParameters,
    
    alias = cms.untracked.string('ANTIKT10CaloJet'),
    FJ_ktRParam = cms.double(1.0)
)


