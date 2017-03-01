import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderTagInfos_cfi import *

pfInclusiveSecondaryVertexFinderCA15TagInfos = pfInclusiveSecondaryVertexFinderTagInfos.clone(
    trackIPTagInfos  = cms.InputTag("pfImpactParameterCA15TagInfos"),
    extSVDeltaRToJet = cms.double(1.5)
)
pfInclusiveSecondaryVertexFinderCA15TagInfos.trackSelection.jetDeltaRMax = cms.double(1.5) # plays no role since using IVF vertices
pfInclusiveSecondaryVertexFinderCA15TagInfos.vertexCuts.maxDeltaRToJetAxis = cms.double(1.5)
