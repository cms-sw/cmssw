import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cfi import *
candidateCombinedSecondaryVertex = cms.ESProducer("CandidateCombinedSecondaryVertexESProducer",
        combinedSecondaryVertexCommon,
        useCategories = cms.bool(True),
        calibrationRecords = cms.vstring(
                'CombinedSVRecoVertex',
                'CombinedSVPseudoVertex',
                'CombinedSVNoVertex'),
        categoryVariableName = cms.string('vertexCategory')
)


#combinedSecondaryVertexV1 = cms.ESProducer("CombinedSecondaryVertexESProducer",
#	combinedSecondaryVertexCommon,
#	useCategories = cms.bool(True),
#	calibrationRecords = cms.vstring(
#		'CombinedSVRetrainRecoVertex', 
#		'CombinedSVRetrainPseudoVertex', 
#		'CombinedSVRetrainNoVertex'),
#	categoryVariableName = cms.string('vertexCategory')
#)

#combinedSecondaryVertexV2 = cms.ESProducer("CombinedSecondaryVertexESProducerV2",
#	combinedSecondaryVertexCommon,
#	useCategories = cms.bool(True),
#	calibrationRecords = cms.vstring(
#		'CombinedSVIVFV2RecoVertex', 
#		'CombinedSVIVFV2PseudoVertex', 
#		'CombinedSVIVFV2NoVertex'),
#	categoryVariableName = cms.string('vertexCategory')
#)
