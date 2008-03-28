import FWCore.ParameterSet.Config as cms

# $Id: iterativeCone5GenJets.cff,v 1.2 2007/05/19 04:20:09 fedor Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.IconeJetParameters_cfi import *
iterativeCone5GenJets = cms.EDProducer("IterativeConeJetProducer",
    GenJetParameters,
    IconeJetParameters,
    alias = cms.untracked.string('IC5GenJet'),
    coneRadius = cms.double(0.5)
)

iterativeCone5GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("iterativeCone5GenJets"),
    ptMin = cms.double(10.0)
)

iterativeCone5GenJetsPt10Seq = cms.Sequence(iterativeCone5GenJets*iterativeCone5GenJetsPt10)

