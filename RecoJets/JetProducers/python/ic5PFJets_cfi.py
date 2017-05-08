import FWCore.ParameterSet.Config as cms

iterativeCone5PFJets = cms.EDProducer("FastjetJetProducer",
	jetAlgorithm = cms.string("IterativeCone"),
	rParam       = cms.double(0.5)
	)

