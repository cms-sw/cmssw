import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderTagInfos_cfi import *

pfInclusiveSecondaryVertexFinderNegativeTagInfos = pfInclusiveSecondaryVertexFinderTagInfos.clone(
    extSVDeltaRToJet = -0.3,
    extSVCollection  = 'inclusiveCandidateNegativeSecondaryVertices',
    vertexCuts = dict(
        distVal2dMin = -2.5,
        distVal2dMax = -0.01,
        distSig2dMin = -99999.9,
        distSig2dMax = -2.0,
        maxDeltaRToJetAxis = -0.5
        )
)
