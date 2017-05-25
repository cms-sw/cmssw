import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.CaloJetParameters_cfi import *

sisCone5CaloJets = cms.EDProducer("FastjetJetProducer",
	CaloJetParameters,
	jetAlgorithm = cms.string("SISCone"),
	rParam       = cms.double(0.5)
	)

