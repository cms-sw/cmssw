import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.TrackJetParameters_cfi import *

kt4TrackJets = cms.EDProducer("FastjetJetProducer",
	TrackJetParameters,
	jetAlgorithm = cms.string("Kt"),
	rParam       = cms.double(0.4)
	)

