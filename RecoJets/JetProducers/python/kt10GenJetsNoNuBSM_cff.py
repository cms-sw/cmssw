import FWCore.ParameterSet.Config as cms

# $Id: kt10GenJetsNoNuBSM.cff,v 1.1 2007/08/02 21:58:23 fedor Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt10GenJetsNoNuBSM = cms.EDProducer("KtJetProducer",
    FastjetNoPU,
    KtJetParameters,
    GenJetParametersNoNuBSM,
    JetPtMin = cms.double(1.0),
    alias = cms.untracked.string('KT10GenJetNoNuBSM'),
    FJ_ktRParam = cms.double(1.0)
)


