import FWCore.ParameterSet.Config as cms

gk5PFJets = cms.EDProducer("FastjetJetProducer",
	jetAlgorithm = cms.string("GeneralizedKt"),
	rParam       = cms.double(0.5)
	)

