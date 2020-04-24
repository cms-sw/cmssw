import FWCore.ParameterSet.Config as cms
from DQMOffline.RecoB.bTagGhostTrackVariables_cff import *

bTagGhostTrackAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        categoryVariable = cms.string('vertexCategory'),
        categories = cms.VPSet(cms.PSet(
            GhostTrackNoVertexVariables,
            GhostTrackPseudoVertexVariables,
            GhostTrackRecoVertexVariables
        ), 
            cms.PSet(
                GhostTrackNoVertexVariables,
                GhostTrackPseudoVertexVariables,
                GhostTrackRecoVertexVariables
            ), 
            cms.PSet(
                GhostTrackNoVertexVariables,
                GhostTrackPseudoVertexVariables
            ), 
            cms.PSet(
                GhostTrackNoVertexVariables
            ))
    )
)

