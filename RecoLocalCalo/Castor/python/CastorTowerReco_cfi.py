import FWCore.ParameterSet.Config as cms

	
CastorTowerReco = cms.EDProducer('CastorTowerProducer',
	inputprocess = cms.untracked.string("castorreco"),
	towercut = cms.untracked.double(1.),
	mintime = cms.untracked.double(-99999.), 
	maxtime = cms.untracked.double(99999.) )

