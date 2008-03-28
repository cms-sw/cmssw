import FWCore.ParameterSet.Config as cms

# $Id: midPointCone5GenJetsNoNuBSM.cff,v 1.1 2007/05/17 23:56:46 fedor Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.MconeJetParameters_cfi import *
midPointCone5GenJetsNoNuBSM = cms.EDProducer("MidpointJetProducer",
    MconeJetParameters,
    GenJetParametersNoNuBSM,
    alias = cms.untracked.string('MC5GenJetNoNuBSM'),
    coneRadius = cms.double(0.5)
)


