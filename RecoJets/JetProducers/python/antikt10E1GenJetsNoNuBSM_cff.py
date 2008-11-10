import FWCore.ParameterSet.Config as cms

# $Id: antikt10E1GenJetsNoNuBSM_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.AntiKtJetParameters_cfi import *
antikt10E1GenJetsNoNuBSM = cms.EDProducer("AntiKtProducer",
    FastjetNoPU,
    AntiKtJetParameters,
    GenJetParametersNoNuBSM,
    
    alias = cms.untracked.string('ANTIKT10E1GenJetNoNuBSM'),
    FJ_ktRParam = cms.double(1.0)
)

antikt10E1GenJetsNoNuBSM.inputEtMin = 0.
antikt10E1GenJetsNoNuBSM.inputEMin = 1.

