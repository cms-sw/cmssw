import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderTagInfos_cfi import *

pfInclusiveSecondaryVertexFinderTagInfosCA15 = pfInclusiveSecondaryVertexFinderTagInfos.clone(
    trackIPTagInfos  = cms.InputTag("pfImpactParameterTagInfosCA15"),
    extSVDeltaRToJet = cms.double(1.5)
)
pfInclusiveSecondaryVertexFinderTagInfosCA15.trackSelection.jetDeltaRMax = cms.double(1.5) # plays no role since using IVF vertices
pfInclusiveSecondaryVertexFinderTagInfosCA15.vertexCuts.maxDeltaRToJetAxis = cms.double(1.5)
