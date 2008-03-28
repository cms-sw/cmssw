import FWCore.ParameterSet.Config as cms

# $Id: iterativeCone7GenJetsNoNuBSM.cff,v 1.1 2007/05/17 23:56:46 fedor Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.IconeJetParameters_cfi import *
iterativeCone7GenJetsNoNuBSM = cms.EDProducer("IterativeConeJetProducer",
    GenJetParametersNoNuBSM,
    IconeJetParameters,
    alias = cms.untracked.string('IC7GenJetNoNuBSM'),
    coneRadius = cms.double(0.7)
)


