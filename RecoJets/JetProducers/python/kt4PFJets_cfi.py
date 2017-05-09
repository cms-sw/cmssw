import FWCore.ParameterSet.Config as cms

kt4PFJets = cms.EDProducer("FastjetJetProducer",
	jetAlgorithm = cms.string("Kt"),
	rParam       = cms.double(0.4)
	)

