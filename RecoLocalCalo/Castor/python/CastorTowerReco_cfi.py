import FWCore.ParameterSet.Config as cms

	
CastorTowerReco = cms.EDProducer('CastorTowerProducer',
	inputprocess = cms.untracked.string("castorreco"),
	towercut = cms.untracked.double(0.65), # 4*sigma cut per channel in GeV
	mintime = cms.untracked.double(-99999.), 
	maxtime = cms.untracked.double(99999.) )

