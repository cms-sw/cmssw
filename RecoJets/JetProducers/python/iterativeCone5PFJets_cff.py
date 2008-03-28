import FWCore.ParameterSet.Config as cms

# $Id: iterativeCone5PFJets.cff,v 1.1 2007/10/06 13:56:59 weng Exp $
from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.IconeJetParameters_cfi import *
iterativeCone5PFJets = cms.EDProducer("IterativeConeJetProducer",
    IconeJetParameters,
    PFJetParameters,
    alias = cms.untracked.string('IC5PFJet'),
    coneRadius = cms.double(0.5)
)


