import FWCore.ParameterSet.Config as cms

# $Id: midPointCone5PFJets.cff,v 1.1 2007/10/06 13:56:59 weng Exp $
from RecoJets.JetProducers.PFJetParameters_cfi import *
from RecoJets.JetProducers.MconeJetParameters_cfi import *
midPointCone5PFJets = cms.EDProducer("MidpointJetProducer",
    MconeJetParameters,
    PFJetParameters,
    alias = cms.untracked.string('MC5PFJet'),
    coneRadius = cms.double(0.5)
)


