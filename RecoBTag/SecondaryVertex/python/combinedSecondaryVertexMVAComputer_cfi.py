import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cff import *

combinedSecondaryVertexMVAComputer = cms.ESProducer("CombinedSecondaryVertexESProducer",
	combinedSecondaryVertexCommon,
	useCategories = cms.bool(True),
	calibrationRecords = cms.vstring(
		'CombinedSVMVARecoVertex', 
		'CombinedSVMVAPseudoVertex', 
		'CombinedSVMVANoVertex'),
	categoryVariableName = cms.string('vertexCategory')
)
