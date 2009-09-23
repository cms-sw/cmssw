import FWCore.ParameterSet.Config as cms

# $Id: antikt4GenJets_cff.py,v 1.1 2008/11/10 19:05:03 srappocc Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.AntiKtJetParameters_cfi import *
antikt7GenJets = cms.EDProducer("AntiKtJetProducer",
    GenJetParameters,
    FastjetNoPU,
    AntiKtJetParameters,
    
    alias = cms.untracked.string('ANTIKT5GenJet'),
    FJ_ktRParam = cms.double(0.7)
)

antikt7GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("antikt7GenJets"),
    ptMin = cms.double(10.0)
)

antikt7GenJetsPt10Seq = cms.Sequence(antikt7GenJets*antikt7GenJetsPt10)

