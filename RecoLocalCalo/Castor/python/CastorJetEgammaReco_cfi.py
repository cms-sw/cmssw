import FWCore.ParameterSet.Config as cms

CastorJetEgammaRecoCustomKt = cms.EDProducer('CastorJetEgammaProducer',
	input = cms.untracked.string("CastorClusterRecoCustomKt"),
	fastsim = cms.untracked.bool(False) )
	
CastorJetEgammaRecoKt = cms.EDProducer('CastorJetEgammaProducer',
	input = cms.untracked.string("CastorClusterRecoKt"),
	fastsim = cms.untracked.bool(False) )
	
CastorJetEgammaRecoSISCone = cms.EDProducer('CastorJetEgammaProducer',
	input = cms.untracked.string("CastorClusterRecoSISCone"),
	fastsim = cms.untracked.bool(False) )
	

