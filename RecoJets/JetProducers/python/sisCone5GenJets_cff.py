import FWCore.ParameterSet.Config as cms

# $Id: sisCone5GenJets.cff,v 1.1 2007/08/02 21:58:23 fedor Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.SISConeJetParameters_cfi import *
sisCone5GenJets = cms.EDProducer("SISConeJetProducer",
    GenJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    alias = cms.untracked.string('SISC5GenJet'),
    coneRadius = cms.double(0.5)
)

sisCone5GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("sisCone5GenJets"),
    ptMin = cms.double(10.0)
)

sisCone5GenJetsPt10Seq = cms.Sequence(sisCone5GenJets*sisCone5GenJetsPt10)

