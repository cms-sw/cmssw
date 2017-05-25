import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.GenJetParameters_cfi import *

gk5GenJets = cms.EDProducer("FastjetJetProducer",
	GenJetParameters,
	jetAlgorithm = cms.string("GeneralizedKt"),
	rParam       = cms.double(0.5)
	)
