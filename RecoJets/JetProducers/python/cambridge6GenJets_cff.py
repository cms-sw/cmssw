import FWCore.ParameterSet.Config as cms

# $Id: cambridge6GenJets_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.CambridgeJetParameters_cfi import *
cambridge6GenJets = cms.EDProducer("CambridgeJetProducer",
    GenJetParameters,
    FastjetNoPU,
    CambridgeJetParameters,
    
    alias = cms.untracked.string('CAMBRIDGE6GenJet'),
    FJ_ktRParam = cms.double(0.6)
)

cambridge6GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("cambridge6GenJets"),
    ptMin = cms.double(10.0)
)

cambridge6GenJetsPt10Seq = cms.Sequence(cambridge6GenJets*cambridge6GenJetsPt10)

