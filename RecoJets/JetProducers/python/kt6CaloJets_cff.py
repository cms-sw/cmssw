import FWCore.ParameterSet.Config as cms

# $Id: kt6CaloJets.cff,v 1.1 2007/08/02 21:58:23 fedor Exp $
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt6CaloJets = cms.EDProducer("KtJetProducer",
    FastjetNoPU,
    KtJetParameters,
    CaloJetParameters,
    JetPtMin = cms.double(1.0),
    alias = cms.untracked.string('KT6CaloJet'),
    FJ_ktRParam = cms.double(0.6)
)


