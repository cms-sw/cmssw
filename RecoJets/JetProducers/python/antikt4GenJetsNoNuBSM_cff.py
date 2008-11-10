import FWCore.ParameterSet.Config as cms

# $Id: antikt4GenJetsNoNuBSM_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.AntiKtJetParameters_cfi import *
antikt4GenJetsNoNuBSM = cms.EDProducer("AntiKtJetProducer",
    FastjetNoPU,
    AntiKtJetParameters,
    GenJetParametersNoNuBSM,
    
    alias = cms.untracked.string('ANTIKT4GenJetNoNuBSM'),
    FJ_ktRParam = cms.double(0.4)
)


