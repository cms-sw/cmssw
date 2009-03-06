import FWCore.ParameterSet.Config as cms

# $Id: cdfMidpointCone5GenJets_cff.py,v 1.3 2008/08/20 16:10:09 oehler Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.MconeJetParameters_cfi import *
cdfMidpointCone5GenJets = cms.EDProducer("CDFMidpointJetProducer",
    MconeJetParameters,
    FastjetNoPU,
    GenJetParameters,
    coneRadius = cms.double(0.5),
    
    alias = cms.untracked.string('MC5GenJet')
)

cdfMidpointCone5GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("cdfMidpointCone5GenJets"),
    ptMin = cms.double(10.0)
)

cdfMidpointCone5GenJetsPt10Seq = cms.Sequence(cdfMidpointCone5GenJets*cdfMidpointCone5GenJetsPt10)

