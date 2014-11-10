import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.ghostTrackCommon_cff import *

ghostTrackComputer = cms.ESProducer("GhostTrackESProducer",
	ghostTrackCommon,
	useCategories = cms.bool(True),
	calibrationRecords = cms.vstring(
		'GhostTrackRecoVertex', 
		'GhostTrackPseudoVertex', 
		'GhostTrackNoVertex'),
	categoryVariableName = cms.string('vertexCategory')
)
