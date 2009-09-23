import FWCore.ParameterSet.Config as cms

# $Id: RecoGenJets_cff.py,v 1.3 2008/08/20 15:58:57 oehler Exp $
#
# ShR 27 Mar 07: move modules producing candidates for Jets into separate cff file due to scheduling problem
#
from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.FastjetParameters_cfi import *
from RecoJets.JetProducers.KtJetParameters_cfi import *
from RecoJets.JetProducers.SISConeJetParameters_cfi import *
from RecoJets.JetProducers.IconeJetParameters_cfi import *
from RecoJets.JetProducers.AntiKtJetParameters_cfi import *

kt4GenJets = cms.EDProducer("KtJetProducer",
    GenJetParameters,
    FastjetNoPU,
    KtJetParameters,
    alias = cms.untracked.string('KT4GenJet'),
    FJ_ktRParam = cms.double(0.4)
)

kt6GenJets = cms.EDProducer("KtJetProducer",
    GenJetParameters,
    FastjetNoPU,
    KtJetParameters,
    alias = cms.untracked.string('KT6GenJet'),
    FJ_ktRParam = cms.double(0.6)
)

iterativeCone5GenJets = cms.EDProducer("IterativeConeJetProducer",
    GenJetParameters,
    IconeJetParameters,
    alias = cms.untracked.string('IC5GenJet'),
    coneRadius = cms.double(0.5)
)

sisCone5GenJets = cms.EDProducer("SISConeJetProducer",
    GenJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    alias = cms.untracked.string('SISC5GenJet'),
    coneRadius = cms.double(0.5)
)

sisCone7GenJets = cms.EDProducer("SISConeJetProducer",
    GenJetParameters,
    SISConeJetParameters,
    FastjetNoPU,
    alias = cms.untracked.string('SISC7GenJet'),
    coneRadius = cms.double(0.7)
)

antikt5GenJets = cms.EDProducer("AntiKtJetProducer",
    GenJetParameters,
    FastjetNoPU,
    AntiKtJetParameters,
    alias = cms.untracked.string('ANTIKT5GenJet'),
    FJ_ktRParam = cms.double(0.5)
)

recoGenJets = cms.Sequence(kt4GenJets+kt6GenJets+antikt5GenJets+iterativeCone5GenJets+sisCone5GenJets+sisCone7GenJets)


