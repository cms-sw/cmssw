import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.ghostTrackCommon_cff import *

candidateGhostTrackComputer = cms.ESProducer("CandidateGhostTrackESProducer",
	ghostTrackCommon,
	useCategories = cms.bool(True),
	calibrationRecords = cms.vstring(
		'GhostTrackRecoVertex', 
		'GhostTrackPseudoVertex', 
		'GhostTrackNoVertex'),
        recordLabel = cms.string(''),
	categoryVariableName = cms.string('vertexCategory')
)
