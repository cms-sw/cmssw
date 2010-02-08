import FWCore.ParameterSet.Config as cms

CastorJetEgammaReco = cms.EDProducer('CastorJetEgammaProducer',
	inputprocess = cms.untracked.string("CastorClusterReco"),
	fastsim = cms.untracked.bool(False),
	KtAlgo = cms.untracked.bool(True),
	ClusterAlgo = cms.untracked.bool(False) )
	

