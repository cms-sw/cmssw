import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.combinedSecondaryVertexCommon_cff import *

candidateCombinedSecondaryVertexComputer = cms.ESProducer("CandidateCombinedSecondaryVertexESProducer",
        combinedSecondaryVertexCommon,
        useCategories = cms.bool(True),
        calibrationRecords = cms.vstring(
                'CombinedSVRecoVertex',
                'CombinedSVPseudoVertex',
                'CombinedSVNoVertex'),
        categoryVariableName = cms.string('vertexCategory')
)
