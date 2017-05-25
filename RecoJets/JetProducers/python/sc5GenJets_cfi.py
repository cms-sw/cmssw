import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.GenJetParameters_cfi import *

sisCone5GenJets = cms.EDProducer("FastjetJetProducer",
	GenJetParameters,
	jetAlgorithm = cms.string("SISCone"),
	rParam       = cms.double(0.5)
	)
