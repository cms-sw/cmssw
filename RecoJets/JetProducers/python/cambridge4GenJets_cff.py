import FWCore.ParameterSet.Config as cms

# $Id: cambridge4GenJets_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.CambridgeJetParameters_cfi import *
cambridge4GenJets = cms.EDProducer("CambridgeJetProducer",
    GenJetParameters,
    FastjetNoPU,
    CambridgeJetParameters,
    
    alias = cms.untracked.string('CAMBRIDGE4GenJet'),
    FJ_ktRParam = cms.double(0.4)
)

cambridge4GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("cambridge4GenJets"),
    ptMin = cms.double(10.0)
)

cambridge4GenJetsPt10Seq = cms.Sequence(cambridge4GenJets*cambridge4GenJetsPt10)

