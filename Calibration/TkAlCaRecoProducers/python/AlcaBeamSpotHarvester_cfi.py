import FWCore.ParameterSet.Config as cms

alcaBeamSpotHarvester = cms.EDAnalyzer("AlcaBeamSpotHarvester",
    AlcaBeamSpotHarvesterParameters = cms.PSet(
	BeamSpotOutputBase = cms.untracked.string("lumibased"), #runbased 
	BeamSpotModuleName = cms.untracked.string("alcaBeamSpotProducer"),
	BeamSpotLabel      = cms.untracked.string("alcaBeamSpot"),
	outputRecordName   = cms.untracked.string("BeamSpotObjectsRcdByLumi"),
	SigmaZValue        = cms.untracked.double(10) 
    ),
    metadataOfflineDropBox = cms.PSet(
        destDB             = cms.untracked.string("oracle://cms_orcoff_prep/CMS_COND_BEAMSPOT"),
        destDBValidation   = cms.untracked.string("oracle://cms_orcoff_prep/CMS_COND_BEAMSPOT"),
        tag                = cms.untracked.string("beamspot_Tier0_offline"),
        DuplicateTagPROMPT = cms.untracked.string("beamspot_Tier0_prompt")
    )
)

