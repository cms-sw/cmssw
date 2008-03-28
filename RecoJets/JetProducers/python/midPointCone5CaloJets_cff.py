import FWCore.ParameterSet.Config as cms

# $Id: midPointCone5CaloJets.cff,v 1.1 2007/05/17 23:56:46 fedor Exp $
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.MconeJetParameters_cfi import *
midPointCone5CaloJets = cms.EDProducer("MidpointJetProducer",
    MconeJetParameters,
    CaloJetParameters,
    alias = cms.untracked.string('MC5CaloJet'),
    coneRadius = cms.double(0.5)
)


