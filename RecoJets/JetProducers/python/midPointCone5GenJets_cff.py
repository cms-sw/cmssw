import FWCore.ParameterSet.Config as cms

# $Id: midPointCone5GenJets.cff,v 1.2 2007/05/19 04:20:09 fedor Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.MconeJetParameters_cfi import *
midPointCone5GenJets = cms.EDProducer("MidpointJetProducer",
    MconeJetParameters,
    GenJetParameters,
    alias = cms.untracked.string('MC5GenJet'),
    coneRadius = cms.double(0.5)
)

midPointCone5GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("midPointCone5GenJets"),
    ptMin = cms.double(10.0)
)

midPointCone5GenJetsPt10Seq = cms.Sequence(midPointCone5GenJets*midPointCone5GenJetsPt10)

