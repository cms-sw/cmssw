import FWCore.ParameterSet.Config as cms

alcaBeamSpotHarvester = cms.EDAnalyzer("AlcaBeamSpotHarvester",
    AlcaBeamSpotHarvesterParameters = cms.PSet(
	BeamSpotOutputBase = cms.untracked.string("lumibased"), #runbased 
	BeamSpotModuleName = cms.untracked.string("alcaBeamSpotProducer"),
	BeamSpotLabel      = cms.untracked.string("alcaBeamSpot"),
	outputRecordName   = cms.untracked.string("BeamSpotObjectsRcdByLumi"),
	SigmaZValue        = cms.untracked.double(-1) 
    )
)

