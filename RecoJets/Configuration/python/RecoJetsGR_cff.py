import FWCore.ParameterSet.Config as cms

# $Id: RecoJetsGR.cff,v 1.1 2008/04/08 20:22:09 fedor Exp $
#
from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.IconeJetParameters_cfi import *
iterativeCone5CaloJets = cms.EDProducer("IterativeConeJetProducer",
    CaloJetParameters,
    IconeJetParameters,
    alias = cms.untracked.string('IC5CaloJet'),
    coneRadius = cms.double(0.5)
)

iterativeCone15CaloJets = cms.EDProducer("IterativeConeJetProducer",
    CaloJetParameters,
    IconeJetParameters,
    alias = cms.untracked.string('IC15CaloJet'),
    seedThreshold = cms.double(0.5),
    coneRadius = cms.double(0.15)
)

recoJetsGR = cms.Sequence(iterativeCone5CaloJets+iterativeCone15CaloJets)

