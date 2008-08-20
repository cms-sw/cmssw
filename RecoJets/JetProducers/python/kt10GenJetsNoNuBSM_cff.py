import FWCore.ParameterSet.Config as cms

# $Id: kt10GenJetsNoNuBSM_cff.py,v 1.2 2008/04/21 03:28:44 rpw Exp $
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


