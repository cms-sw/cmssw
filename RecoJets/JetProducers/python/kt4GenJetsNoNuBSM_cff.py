import FWCore.ParameterSet.Config as cms

# $Id: kt4GenJetsNoNuBSM_cff.py,v 1.2 2008/04/21 03:28:50 rpw Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt4GenJetsNoNuBSM = cms.EDProducer("KtJetProducer",
    FastjetNoPU,
    KtJetParameters,
    GenJetParametersNoNuBSM,
    
    alias = cms.untracked.string('KT4GenJetNoNuBSM'),
    FJ_ktRParam = cms.double(0.4)
)


