import FWCore.ParameterSet.Config as cms

	
CastorTowerCandidateReco = cms.EDProducer('CastorTowerCandidateProducer',
	src = cms.InputTag("CastorTowerReco"),
	minimumE = cms.double(0.) )

