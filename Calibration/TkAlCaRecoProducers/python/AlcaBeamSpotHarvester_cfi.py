import FWCore.ParameterSet.Config as cms

alcaBeamSpotHarvester = cms.EDAnalyzer("AlcaBeamSpotHarvester",
    AlcaBeamSpotHarvesterParameters = cms.PSet(
	BeamSpotOutputBase = cms.untracked.string("runbased") #lumibased 
    )
)

