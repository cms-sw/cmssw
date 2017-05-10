import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.GenJetParameters_cfi import *

ca4GenJets = cms.EDProducer("FastjetJetProducer",
	GenJetParameters,
	jetAlgorithm = cms.string("CambridgeAachen"),
	rParam       = cms.double(0.4)
	)
