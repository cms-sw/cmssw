import FWCore.ParameterSet.Config as cms
	
CastorJetEgammaRecoAntiKt07 = cms.EDProducer('CastorJetEgammaProducer',
	input = cms.untracked.string("CastorClusterRecoAntiKt07"),
	fastsim = cms.untracked.bool(False) )	

