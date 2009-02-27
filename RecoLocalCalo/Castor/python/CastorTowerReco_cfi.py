import FWCore.ParameterSet.Config as cms

	
CastorTowerReco = cms.EDProducer('CastorTowerProducer',
	inputprocess = cms.untracked.string("CastorCellReco"),
	towercut = cms.untracked.double(1.) )

