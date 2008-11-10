import FWCore.ParameterSet.Config as cms

# $Id: cambridge10GenJets_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.CambridgeJetParameters_cfi import *
cambridge10GenJets = cms.EDProducer("CambridgeJetProducer",
    GenJetParameters,
    FastjetNoPU,
    CambridgeJetParameters,
    
    alias = cms.untracked.string('CAMBRIDGE10GenJet'),
    FJ_ktRParam = cms.double(1.0)
)


