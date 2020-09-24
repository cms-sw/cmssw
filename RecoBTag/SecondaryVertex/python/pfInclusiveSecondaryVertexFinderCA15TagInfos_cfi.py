import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderTagInfos_cfi import *

pfInclusiveSecondaryVertexFinderCA15TagInfos = pfInclusiveSecondaryVertexFinderTagInfos.clone(
    trackIPTagInfos  = "pfImpactParameterCA15TagInfos",
    extSVDeltaRToJet = 1.5,
    trackSelection = dict(jetDeltaRMax = 1.5), # plays no role since using IVF vertices
    vertexCuts = dict(maxDeltaRToJetAxis = 1.5)
)
