import FWCore.ParameterSet.Config as cms

# $Id: sisCone7GenJets.cff,v 1.1 2007/08/02 21:58:23 fedor Exp $
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.SISConeJetParameters_cfi import *
sisCone7GenJets = cms.EDProducer("SISConeJetProducer",
    GenJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    alias = cms.untracked.string('SISC7GenJet'),
    coneRadius = cms.double(0.7)
)

sisCone7GenJetsPt10 = cms.EDFilter("PtMinGenJetSelector",
    src = cms.InputTag("sisCone7GenJets"),
    ptMin = cms.double(10.0)
)

sisCone7GenJetsPt10Seq = cms.Sequence(sisCone7GenJets*sisCone7GenJetsPt10)

