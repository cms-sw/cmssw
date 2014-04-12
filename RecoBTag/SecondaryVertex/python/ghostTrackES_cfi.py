import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.ghostTrackCommon_cfi import *

ghostTrack = cms.ESProducer("GhostTrackESProducer",
	ghostTrackCommon,
	useCategories = cms.bool(True),
	calibrationRecords = cms.vstring(
		'GhostTrackRecoVertex', 
		'GhostTrackPseudoVertex', 
		'GhostTrackNoVertex'),
	categoryVariableName = cms.string('vertexCategory')
)
