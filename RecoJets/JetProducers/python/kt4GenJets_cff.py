import FWCore.ParameterSet.Config as cms

# $Id: kt4GenJets.cff,v 1.1 2007/10/26 22:29:55 fedor Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt4GenJets = cms.EDProducer("KtJetProducer",
    GenJetParameters,
    FastjetNoPU,
    KtJetParameters,
    JetPtMin = cms.double(1.0),
    alias = cms.untracked.string('KT4GenJet'),
    FJ_ktRParam = cms.double(0.4)
)

kt4GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("kt4GenJets"),
    ptMin = cms.double(10.0)
)

kt4GenJetsPt10Seq = cms.Sequence(kt4GenJets*kt4GenJetsPt10)

