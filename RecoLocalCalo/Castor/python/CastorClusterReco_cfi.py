import FWCore.ParameterSet.Config as cms

CastorClusterReco = cms.EDProducer('CastorClusterProducer',
	inputprocess = cms.untracked.string("CastorTowerReco"),
	KtAlgo = cms.untracked.bool(True),
	ClusterAlgo = cms.untracked.bool(False),
	KtRecombination = cms.untracked.uint32(2),
	KtrParameter = cms.untracked.double(1.) )
	

