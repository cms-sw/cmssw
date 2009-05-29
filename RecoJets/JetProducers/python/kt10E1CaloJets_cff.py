import FWCore.ParameterSet.Config as cms

# $Id: kt10E1CaloJets_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
kt10E1CaloJets = cms.EDProducer("KtJetProducer",
    FastjetNoPU,
    KtJetParameters,
    CaloJetParameters,
    
    alias = cms.untracked.string('KT10E1CaloJet'),
    FJ_ktRParam = cms.double(1.0)
)

kt10E1CaloJets.inputEtMin = 0.
kt10E1CaloJets.inputEMin = 1.

