import FWCore.ParameterSet.Config as cms

CastorClusterRecoCustomKt = cms.EDProducer('CastorClusterProducer',
	inputtowers = cms.untracked.string("CastorTowerReco"),
	basicjets = cms.untracked.string(""),
	ClusterAlgo = cms.untracked.bool(False) )

CastorClusterRecoAntiKt07 = cms.EDProducer('CastorClusterProducer',
	inputtowers = cms.untracked.string(""),
	basicjets = cms.untracked.string("CastorFastjetRecoAntiKt07"))

