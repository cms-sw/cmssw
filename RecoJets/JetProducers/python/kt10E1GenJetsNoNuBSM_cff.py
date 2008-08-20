import FWCore.ParameterSet.Config as cms

# $Id: kt10E1GenJetsNoNuBSM_cff.py,v 1.2 2008/04/21 03:28:41 rpw Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt10E1GenJetsNoNuBSM = cms.EDProducer("KtProducer",
    FastjetNoPU,
    KtJetParameters,
    GenJetParametersNoNuBSM,
    
    alias = cms.untracked.string('KT10E1GenJetNoNuBSM'),
    FJ_ktRParam = cms.double(1.0)
)

kt10E1GenJetsNoNuBSM.inputEtMin = 0.
kt10E1GenJetsNoNuBSM.inputEMin = 1.

