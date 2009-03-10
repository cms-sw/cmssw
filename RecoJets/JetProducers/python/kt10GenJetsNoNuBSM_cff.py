import FWCore.ParameterSet.Config as cms

# $Id: kt10GenJetsNoNuBSM_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt10GenJetsNoNuBSM = cms.EDProducer("KtJetProducer",
    FastjetNoPU,
    KtJetParameters,
    GenJetParametersNoNuBSM,
    
    alias = cms.untracked.string('KT10GenJetNoNuBSM'),
    FJ_ktRParam = cms.double(1.0)
)


