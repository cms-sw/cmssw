import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.GenJetParameters_cfi import *

kt4GenJets = cms.EDProducer("FastjetJetProducer",
	GenJetParameters,
	jetAlgorithm = cms.string("Kt"),
	rParam       = cms.double(0.4)
	)
