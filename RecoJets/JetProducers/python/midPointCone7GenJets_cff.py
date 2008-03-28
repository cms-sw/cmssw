import FWCore.ParameterSet.Config as cms

# $Id: midPointCone7GenJets.cff,v 1.2 2007/05/19 04:20:09 fedor Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.MconeJetParameters_cfi import *
midPointCone7GenJets = cms.EDProducer("MidpointJetProducer",
    MconeJetParameters,
    GenJetParameters,
    alias = cms.untracked.string('MC7GenJet'),
    coneRadius = cms.double(0.7)
)

midPointCone7GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("midPointCone7GenJets"),
    ptMin = cms.double(10.0)
)

midPointCone7GenJetsPt10Seq = cms.Sequence(midPointCone7GenJets*midPointCone7GenJetsPt10)

