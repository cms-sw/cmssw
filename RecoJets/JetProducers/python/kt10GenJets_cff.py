import FWCore.ParameterSet.Config as cms

# $Id: kt10GenJets_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt10GenJets = cms.EDProducer("KtJetProducer",
    GenJetParameters,
    FastjetNoPU,
    KtJetParameters,
    
    alias = cms.untracked.string('KT10GenJet'),
    FJ_ktRParam = cms.double(1.0)
)


