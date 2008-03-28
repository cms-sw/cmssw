import FWCore.ParameterSet.Config as cms

# $Id: iterativeCone7GenJets.cff,v 1.2 2007/05/19 04:20:09 fedor Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.IconeJetParameters_cfi import *
iterativeCone7GenJets = cms.EDProducer("IterativeConeJetProducer",
    GenJetParameters,
    IconeJetParameters,
    alias = cms.untracked.string('IC7GenJet'),
    coneRadius = cms.double(0.7)
)

iterativeCone7GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("iterativeCone7GenJets"),
    ptMin = cms.double(10.0)
)

iterativeCone7GenJetsPt10Seq = cms.Sequence(iterativeCone7GenJets*iterativeCone7GenJetsPt10)

