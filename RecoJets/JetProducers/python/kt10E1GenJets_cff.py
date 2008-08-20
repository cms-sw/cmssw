import FWCore.ParameterSet.Config as cms

# $Id: kt10E1GenJets_cff.py,v 1.3 2008/07/16 03:37:08 fedor Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt10E1GenJets = cms.EDProducer("KtJetProducer",
    FastjetNoPU,
    KtJetParameters,
    GenJetParameters,
    
    alias = cms.untracked.string('KT10E1GenJet'),
    FJ_ktRParam = cms.double(1.0)
)

kt10E1GenJets.inputEtMin = 0.
kt10E1GenJets.inputEMin = 1.

