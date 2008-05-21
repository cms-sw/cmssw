import FWCore.ParameterSet.Config as cms

# combinedSecondaryVertex jet tag computer configuration
bTagCombinedSVAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        categoryVariable = cms.string('vertexCategory'),
        categories = cms.VPSet(cms.PSet(
            combinedSVNoVertexVariables,
            combinedSVPseudoVertexVariables,
            combinedSVRecoVertexVariables
        ), 
            cms.PSet(
                combinedSVNoVertexVariables,
                combinedSVPseudoVertexVariables,
                combinedSVRecoVertexVariables
            ), 
            cms.PSet(
                combinedSVNoVertexVariables,
                combinedSVPseudoVertexVariables
            ), 
            cms.PSet(
                combinedSVNoVertexVariables
            ))
    )
)


