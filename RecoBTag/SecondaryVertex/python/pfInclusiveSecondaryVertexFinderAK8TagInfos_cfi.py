import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderTagInfos_cfi import *

pfInclusiveSecondaryVertexFinderAK8TagInfos = pfInclusiveSecondaryVertexFinderTagInfos.clone(
    trackIPTagInfos  = "pfImpactParameterAK8TagInfos",
    extSVDeltaRToJet = 0.8,
    trackSelection = dict(jetDeltaRMax = 0.8), # plays no role since using IVF vertices
    vertexCuts = dict(maxDeltaRToJetAxis = 0.8)
)
