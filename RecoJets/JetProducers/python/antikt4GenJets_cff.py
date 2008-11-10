import FWCore.ParameterSet.Config as cms

# $Id: antikt4GenJets_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.AntiKtJetParameters_cfi import *
antikt4GenJets = cms.EDProducer("AntiKtJetProducer",
    GenJetParameters,
    FastjetNoPU,
    AntiKtJetParameters,
    
    alias = cms.untracked.string('ANTIKT4GenJet'),
    FJ_ktRParam = cms.double(0.4)
)

antikt4GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("antikt4GenJets"),
    ptMin = cms.double(10.0)
)

antikt4GenJetsPt10Seq = cms.Sequence(antikt4GenJets*antikt4GenJetsPt10)

