import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFClusterJetParameters_cfi import *

ak5PFClusterJets = cms.EDProducer("FastjetJetProducer",
	PFClusterJetParameters,
	jetAlgorithm = cms.string("AntiKt"),
	rParam       = cms.double(0.5)
	)

