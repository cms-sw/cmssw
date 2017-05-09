import FWCore.ParameterSet.Config as cms

ak5PFJets = cms.EDProducer("FastjetJetProducer",
	jetAlgorithm = cms.string("AntiKt"),
	rParam       = cms.double(0.5)
	)

