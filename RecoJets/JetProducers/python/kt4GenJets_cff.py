import FWCore.ParameterSet.Config as cms

# $Id: kt4GenJets_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt4GenJets = cms.EDProducer("KtJetProducer",
    GenJetParameters,
    FastjetNoPU,
    KtJetParameters,
    
    alias = cms.untracked.string('KT4GenJet'),
    FJ_ktRParam = cms.double(0.4)
)

kt4GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("kt4GenJets"),
    ptMin = cms.double(10.0)
)

kt4GenJetsPt10Seq = cms.Sequence(kt4GenJets*kt4GenJetsPt10)

