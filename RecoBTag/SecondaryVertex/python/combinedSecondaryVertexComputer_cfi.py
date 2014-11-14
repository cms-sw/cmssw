import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cff import *

combinedSecondaryVertexComputer = cms.ESProducer("CombinedSecondaryVertexESProducer",
	combinedSecondaryVertexCommon,
	useCategories = cms.bool(True),
	calibrationRecords = cms.vstring(
		'CombinedSVRecoVertex', 
		'CombinedSVPseudoVertex', 
		'CombinedSVNoVertex'),
	categoryVariableName = cms.string('vertexCategory')
)
