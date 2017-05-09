import FWCore.ParameterSet.Config as cms

sisCone5PFJets = cms.EDProducer("FastjetJetProducer",
	jetAlgorithm = cms.string("SISCone"),
	rParam       = cms.double(0.5)
	)

