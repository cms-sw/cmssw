import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.TrackJetParameters_cfi import *

ak5TrackJets = cms.EDProducer("FastjetJetProducer",
	TrackJetParameters,
	jetAlgorithm = cms.string("AntiKt"),
	rParam       = cms.double(0.5)
	)

