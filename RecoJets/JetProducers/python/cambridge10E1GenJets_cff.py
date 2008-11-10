import FWCore.ParameterSet.Config as cms

# $Id: cambridge10E1GenJets_cff.py,v 1.4 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.CambridgeJetParameters_cfi import *
cambridge10E1GenJets = cms.EDProducer("CambridgeJetProducer",
    FastjetNoPU,
    CambridgeJetParameters,
    GenJetParameters,
    
    alias = cms.untracked.string('CAMBRIDGE10E1GenJet'),
    FJ_ktRParam = cms.double(1.0)
)

cambridge10E1GenJets.inputEtMin = 0.
cambridge10E1GenJets.inputEMin = 1.

