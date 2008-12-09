import FWCore.ParameterSet.Config as cms

CastorFullReco = cms.EDProducer('Castor',
	FullReco = cms.untracked.bool(False),
	KtRecombination = cms.untracked.uint32(2),
	KtrParameter = cms.untracked.double(1.),
	Egamma_minRatio = cms.untracked.double(0.5),
	Egamma_maxWidth = cms.untracked.double(0.2),
	Egamma_maxDepth = cms.untracked.double(14488),
        towercut = cms.untracked.double(0.) )
