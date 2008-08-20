import FWCore.ParameterSet.Config as cms

# $Id: kt4CaloJets_cff.py,v 1.2 2008/04/21 03:28:47 rpw Exp $
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt4CaloJets = cms.EDProducer("KtJetProducer",
    FastjetNoPU,
    KtJetParameters,
    CaloJetParameters,
    
    alias = cms.untracked.string('KT4CaloJet'),
    FJ_ktRParam = cms.double(0.4)
)


