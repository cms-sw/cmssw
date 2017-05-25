import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.TrackJetParameters_cfi import *

sisCone5TrackJets = cms.EDProducer("FastjetJetProducer",
	TrackJetParameters,
	jetAlgorithm = cms.string("SISCone"),
	rParam       = cms.double(0.5)
	)

