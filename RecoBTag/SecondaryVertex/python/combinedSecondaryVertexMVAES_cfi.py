import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cfi import *

combinedSecondaryVertexMVA = cms.ESProducer("CombinedSecondaryVertexESProducer",
	combinedSecondaryVertexCommon,
	useCategories = cms.bool(True),
	calibrationRecords = cms.vstring(
		'CombinedSVMVARecoVertex', 
		'CombinedSVMVAPseudoVertex', 
		'CombinedSVMVANoVertex'),
	categoryVariableName = cms.string('vertexCategory')
)
