import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.CaloJetParameters_cfi import *

kt4CaloJets = cms.EDProducer("FastjetJetProducer",
	CaloJetParameters,
	jetAlgorithm = cms.string("Kt"),
	rParam       = cms.double(0.4)
	)

