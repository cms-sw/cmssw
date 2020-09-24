import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.secondaryVertexTagInfos_cfi import *

secondaryVertexNegativeTagInfos = secondaryVertexTagInfos.clone(
    vertexCuts = dict(distVal2dMin = -2.5,
                      distVal2dMax = -0.01,
                      distSig2dMin = -99999.9,
                      distSig2dMax = -3.0,
                      maxDeltaRToJetAxis = -0.5)
)
