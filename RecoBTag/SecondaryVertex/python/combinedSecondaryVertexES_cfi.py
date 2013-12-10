import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cfi import *

combinedSecondaryVertex = cms.ESProducer("CombinedSecondaryVertexESProducer",
	combinedSecondaryVertexCommon,
	useCategories = cms.bool(True),
	calibrationRecords = cms.vstring(
		'CombinedSVRecoVertex', 
		'CombinedSVPseudoVertex', 
		'CombinedSVNoVertex'),
	categoryVariableName = cms.string('vertexCategory')
)

combinedSecondaryVertexV1 = cms.ESProducer("CombinedSecondaryVertexESProducer",
	combinedSecondaryVertexCommon,
	useCategories = cms.bool(True),
	calibrationRecords = cms.vstring(
		'CombinedSVRetrainRecoVertex', 
		'CombinedSVRetrainPseudoVertex', 
		'CombinedSVRetrainNoVertex'),
	categoryVariableName = cms.string('vertexCategory')
)
