import FWCore.ParameterSet.Config as cms

# $Id: BasicJetIcone5.cfi,v 1.3 2007/02/08 01:46:11 fedor Exp $
from RecoJets.JetProducers.IconeJetParameters_cfi import *
iterativeCone5BasicJets = cms.EDProducer("IterativeConeJetProducer",
    IconeJetParameters,
    src = cms.InputTag("caloTowers"),
    coneRadius = cms.double(0.5),
    inputEtMin = cms.double(0.0),
    alias = cms.untracked.string('IC5CaloJet'),
    inputEMin = cms.double(0.0),
    jetType = cms.untracked.string('BasicJet'),
    towerThreshold = cms.double(0.5)
)


