import FWCore.ParameterSet.Config as cms

ca4PFJets = cms.EDProducer("FastjetJetProducer",
	jetAlgorithm = cms.string("CambridgeAachen"),
	rParam       = cms.double(0.4)
	)

