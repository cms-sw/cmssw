import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.pfSecondaryVertexTagInfos_cfi import *
from RecoBTag.SecondaryVertex.ghostTrackVertexReco_cff import *

pfGhostTrackVertexTagInfos = pfSecondaryVertexTagInfos.clone(
    vertexReco = ghostTrackVertexRecoBlock.vertexReco,
    vertexCuts = dict(multiplicityMin = 1)
)
