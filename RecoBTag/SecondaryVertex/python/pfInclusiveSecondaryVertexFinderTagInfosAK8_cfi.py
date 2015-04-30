import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderTagInfos_cfi import *

pfInclusiveSecondaryVertexFinderTagInfosAK8 = pfInclusiveSecondaryVertexFinderTagInfos.clone(
    trackIPTagInfos  = cms.InputTag("pfImpactParameterTagInfosAK8"),
    extSVDeltaRToJet = cms.double(0.8)
)
pfInclusiveSecondaryVertexFinderTagInfosAK8.trackSelection.jetDeltaRMax = cms.double(0.8) # plays no role since using IVF vertices
pfInclusiveSecondaryVertexFinderTagInfosAK8.vertexCuts.maxDeltaRToJetAxis = cms.double(0.8)
