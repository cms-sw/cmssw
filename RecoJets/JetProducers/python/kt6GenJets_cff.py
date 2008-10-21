import FWCore.ParameterSet.Config as cms

# $Id: kt6GenJets.cff,v 1.1 2007/08/02 21:58:23 fedor Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt6GenJets = cms.EDProducer("KtJetProducer",
    GenJetParameters,
    FastjetNoPU,
    KtJetParameters,
    JetPtMin = cms.double(1.0),
    alias = cms.untracked.string('KT6GenJet'),
    FJ_ktRParam = cms.double(0.6)
)

kt6GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("kt6GenJets"),
    ptMin = cms.double(10.0)
)

kt6GenJetsPt10Seq = cms.Sequence(kt6GenJets*kt6GenJetsPt10)

