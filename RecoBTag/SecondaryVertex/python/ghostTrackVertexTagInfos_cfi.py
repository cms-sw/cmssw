import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.secondaryVertexTagInfos_cfi import *
from RecoBTag.SecondaryVertex.ghostTrackVertexReco_cfi import *

ghostTrackVertexTagInfos = secondaryVertexTagInfos.clone()
ghostTrackVertexTagInfos.vertexReco = ghostTrackVertexRecoBlock.vertexReco
ghostTrackVertexTagInfos.vertexCuts.multiplicityMin = 1
