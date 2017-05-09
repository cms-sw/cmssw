import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.TrackJetParameters_cfi import *

ca4TrackJets = cms.EDProducer("FastjetJetProducer",
	TrackJetParameters,
	jetAlgorithm = cms.string("CambridgeAachen"),
	rParam       = cms.double(0.4)
	)

