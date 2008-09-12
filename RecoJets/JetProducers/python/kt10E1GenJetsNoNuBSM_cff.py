import FWCore.ParameterSet.Config as cms

# $Id: kt10E1GenJetsNoNuBSM.cff,v 1.1 2007/08/02 21:58:22 fedor Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt10E1GenJetsNoNuBSM = cms.EDProducer("KtProducer",
    FastjetNoPU,
    KtJetParameters,
    GenJetParametersNoNuBSM,
    JetPtMin = cms.double(1.0),
    alias = cms.untracked.string('KT10E1GenJetNoNuBSM'),
    FJ_ktRParam = cms.double(1.0)
)

kt10E1GenJetsNoNuBSM.inputEtMin = 0.
kt10E1GenJetsNoNuBSM.inputEMin = 1.

