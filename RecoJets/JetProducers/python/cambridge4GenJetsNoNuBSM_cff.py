import FWCore.ParameterSet.Config as cms

# $Id: cambridge4GenJetsNoNuBSM_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParametersNoNuBSM_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.CambridgeJetParameters_cfi import *
cambridge4GenJetsNoNuBSM = cms.EDProducer("CambridgeJetProducer",
    FastjetNoPU,
    CambridgeJetParameters,
    GenJetParametersNoNuBSM,
    
    alias = cms.untracked.string('CAMBRIDGE4GenJetNoNuBSM'),
    FJ_ktRParam = cms.double(0.4)
)


