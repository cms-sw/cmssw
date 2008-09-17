import FWCore.ParameterSet.Config as cms

# $Id: cdfMidpointCone5GenJets.cff,v 1.1 2007/08/02 21:58:22 fedor Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.MconeJetParameters_cfi import *
cdfMidpointCone5GenJets = cms.EDProducer("CDFMidpointJetProducer",
    MconeJetParameters,
    FastjetNoPU,
    GenJetParameters,
    coneRadius = cms.double(0.5),
    JetPtMin = cms.double(0.0),
    alias = cms.untracked.string('MC5GenJet')
)

cdfMidpointCone5GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("cdfMidpointCone5GenJets"),
    ptMin = cms.double(10.0)
)

cdfMidpointCone5GenJetsPt10Seq = cms.Sequence(cdfMidpointCone5GenJets*cdfMidpointCone5GenJetsPt10)

