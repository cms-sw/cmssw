import FWCore.ParameterSet.Config as cms


CastorCellReco = cms.EDProducer('CastorCellProducer',
	inputprocess = cms.untracked.string("castorreco") )
	

