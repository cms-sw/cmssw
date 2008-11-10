import FWCore.ParameterSet.Config as cms

# $Id: cambridge10CaloJets_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.CambridgeJetParameters_cfi import *
cambridge10CaloJets = cms.EDProducer("CambridgeJetProducer",
    FastjetNoPU,
    CambridgeJetParameters,
    CaloJetParameters,
    
    alias = cms.untracked.string('CAMBRIDGE10CaloJet'),
    FJ_ktRParam = cms.double(1.0)
)


