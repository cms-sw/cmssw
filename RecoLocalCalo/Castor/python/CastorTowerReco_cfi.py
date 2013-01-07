import FWCore.ParameterSet.Config as cms

	
CastorTowerReco = cms.EDProducer('CastorTowerProducer',
	inputprocess = cms.string("castorreco"),
	towercut = cms.double(0.65), # 4*sigma cut per channel in GeV
	mintime = cms.double(-99999.), 
	maxtime = cms.double(99999.) )

