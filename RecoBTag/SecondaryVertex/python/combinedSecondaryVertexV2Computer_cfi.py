import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cff import *

combinedSecondaryVertexV2Computer = cms.ESProducer("CombinedSecondaryVertexESProducer",
	combinedSecondaryVertexCommon,
	useCategories = cms.bool(True),
	calibrationRecords = cms.vstring(
		'CombinedSVIVFV2RecoVertex', 
		'CombinedSVIVFV2PseudoVertex', 
		'CombinedSVIVFV2NoVertex'),
	categoryVariableName = cms.string('vertexCategory')
)
