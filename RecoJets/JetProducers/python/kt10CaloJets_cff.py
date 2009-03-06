import FWCore.ParameterSet.Config as cms

# $Id: kt10CaloJets.cff,v 1.1 2007/08/02 21:58:22 fedor Exp $
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt10CaloJets = cms.EDProducer("KtJetProducer",
    FastjetNoPU,
    KtJetParameters,
    CaloJetParameters,
    JetPtMin = cms.double(1.0),
    alias = cms.untracked.string('KT10CaloJet'),
    FJ_ktRParam = cms.double(1.0)
)


