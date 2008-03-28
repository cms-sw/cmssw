import FWCore.ParameterSet.Config as cms

# $Id: kt4CaloJets.cff,v 1.1 2007/10/26 22:29:55 fedor Exp $
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt4CaloJets = cms.EDProducer("KtJetProducer",
    FastjetNoPU,
    KtJetParameters,
    CaloJetParameters,
    JetPtMin = cms.double(1.0),
    alias = cms.untracked.string('KT4CaloJet'),
    FJ_ktRParam = cms.double(0.4)
)


