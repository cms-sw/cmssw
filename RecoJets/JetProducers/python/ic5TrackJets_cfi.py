import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.TrackJetParameters_cfi import *

iterativeCone5TrackJets = cms.EDProducer("FastjetJetProducer",
	TrackJetParameters,
	jetAlgorithm = cms.string("IterativeCone"),
	rParam       = cms.double(0.5)
	)

