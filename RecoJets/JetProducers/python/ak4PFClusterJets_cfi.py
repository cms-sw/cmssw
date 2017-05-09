import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.PFClusterJetParameters_cfi import *

ak4PFClusterJets = cms.EDProducer("FastjetJetProducer",
	PFClusterJetParameters,
	jetAlgorithm = cms.string("AntiKt"),
	rParam       = cms.double(0.4)
	)

