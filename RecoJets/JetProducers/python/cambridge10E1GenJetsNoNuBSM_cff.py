import FWCore.ParameterSet.Config as cms

# $Id: cambridge10E1GenJetsNoNuBSM_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.CambridgeJetParameters_cfi import *
cambridge10E1GenJetsNoNuBSM = cms.EDProducer("CambridgeProducer",
    FastjetNoPU,
    CambridgeJetParameters,
    GenJetParametersNoNuBSM,
    
    alias = cms.untracked.string('CAMBRIDGE10E1GenJetNoNuBSM'),
    FJ_ktRParam = cms.double(1.0)
)

cambridge10E1GenJetsNoNuBSM.inputEtMin = 0.
cambridge10E1GenJetsNoNuBSM.inputEMin = 1.

