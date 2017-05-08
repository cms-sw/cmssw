import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.CaloJetParameters_cfi import *

gk5CaloJets = cms.EDProducer("FastjetJetProducer",
	CaloJetParameters,
	jetAlgorithm = cms.string("GeneralizedKt"),
	rParam       = cms.double(0.5)
	)

