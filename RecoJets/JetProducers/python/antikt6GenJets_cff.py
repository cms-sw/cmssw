import FWCore.ParameterSet.Config as cms

# $Id: antikt6GenJets_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.AntiKtJetParameters_cfi import *
antikt6GenJets = cms.EDProducer("AntiKtJetProducer",
    GenJetParameters,
    FastjetNoPU,
    AntiKtJetParameters,
    
    alias = cms.untracked.string('ANTIKT6GenJet'),
    FJ_ktRParam = cms.double(0.6)
)

antikt6GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("antikt6GenJets"),
    ptMin = cms.double(10.0)
)

antikt6GenJetsPt10Seq = cms.Sequence(antikt6GenJets*antikt6GenJetsPt10)

