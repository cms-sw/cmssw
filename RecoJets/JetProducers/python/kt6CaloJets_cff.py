import FWCore.ParameterSet.Config as cms

# $Id: kt6CaloJets_cff.py,v 1.2 2008/04/21 03:28:51 rpw Exp $
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt6CaloJets = cms.EDProducer("KtJetProducer",
    FastjetNoPU,
    KtJetParameters,
    CaloJetParameters,
    
    alias = cms.untracked.string('KT6CaloJet'),
    FJ_ktRParam = cms.double(0.6)
)


