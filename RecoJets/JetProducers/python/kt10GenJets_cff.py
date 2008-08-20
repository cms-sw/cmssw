import FWCore.ParameterSet.Config as cms

# $Id: kt10GenJets_cff.py,v 1.2 2008/04/21 03:28:43 rpw Exp $
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


