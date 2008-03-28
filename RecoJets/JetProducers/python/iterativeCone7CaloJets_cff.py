import FWCore.ParameterSet.Config as cms

# $Id: iterativeCone7CaloJets.cff,v 1.1 2007/05/17 23:56:46 fedor Exp $
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.IconeJetParameters_cfi import *
iterativeCone7CaloJets = cms.EDProducer("IterativeConeJetProducer",
    CaloJetParameters,
    IconeJetParameters,
    alias = cms.untracked.string('IC7CaloJet'),
    coneRadius = cms.double(0.7)
)


