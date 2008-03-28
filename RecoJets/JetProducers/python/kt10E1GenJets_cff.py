import FWCore.ParameterSet.Config as cms

# $Id: kt10E1GenJets.cff,v 1.1 2007/08/02 21:58:22 fedor Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt10E1GenJets = cms.EDProducer("KtJetProducer",
    FastjetNoPU,
    KtJetParameters,
    CaloJetParameters,
    JetPtMin = cms.double(1.0),
    alias = cms.untracked.string('KT10E1GenJet'),
    FJ_ktRParam = cms.double(1.0)
)

kt10E1GenJets.inputEtMin = 0.
kt10E1GenJets.inputEMin = 1.

