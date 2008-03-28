import FWCore.ParameterSet.Config as cms

# $Id: kt10GenJets.cff,v 1.1 2007/08/02 21:58:22 fedor Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt10GenJets = cms.EDProducer("KtJetProducer",
    GenJetParameters,
    FastjetNoPU,
    KtJetParameters,
    JetPtMin = cms.double(1.0),
    alias = cms.untracked.string('KT10GenJet'),
    FJ_ktRParam = cms.double(1.0)
)


