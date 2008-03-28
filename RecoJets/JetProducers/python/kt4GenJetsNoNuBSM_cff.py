import FWCore.ParameterSet.Config as cms

# $Id: kt4GenJetsNoNuBSM.cff,v 1.1 2007/10/26 22:29:55 fedor Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt4GenJetsNoNuBSM = cms.EDProducer("KtJetProducer",
    FastjetNoPU,
    KtJetParameters,
    GenJetParametersNoNuBSM,
    JetPtMin = cms.double(1.0),
    alias = cms.untracked.string('KT4GenJetNoNuBSM'),
    FJ_ktRParam = cms.double(0.4)
)


