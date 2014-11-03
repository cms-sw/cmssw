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

candidateCombinedSecondaryVertexV2 = cms.ESProducer("CandidateCombinedSecondaryVertexESProducerV2",
        combinedSecondaryVertexCommon,
        useCategories = cms.bool(True),
        calibrationRecords = cms.vstring(
               'CombinedSVIVFV2RecoVertex', 
               'CombinedSVIVFV2PseudoVertex', 
               'CombinedSVIVFV2NoVertex'),
        categoryVariableName = cms.string('vertexCategory')
)

