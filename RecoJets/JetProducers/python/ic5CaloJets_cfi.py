import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.CaloJetParameters_cfi import *

iterativeCone5CaloJets = cms.EDProducer("FastjetJetProducer",
	CaloJetParameters,
	jetAlgorithm = cms.string("IterativeCone"),
	rParam       = cms.double(0.5)
	)

