import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.CaloJetParameters_cfi import *

ak5CaloJets = cms.EDProducer("FastjetJetProducer",
	CaloJetParameters,
	jetAlgorithm = cms.string("AntiKt"),
	rParam       = cms.double(0.5)
	)

