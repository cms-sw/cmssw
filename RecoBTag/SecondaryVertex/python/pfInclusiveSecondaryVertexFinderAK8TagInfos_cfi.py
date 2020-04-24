import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.pfInclusiveSecondaryVertexFinderTagInfos_cfi import *

pfInclusiveSecondaryVertexFinderAK8TagInfos = pfInclusiveSecondaryVertexFinderTagInfos.clone(
    trackIPTagInfos  = cms.InputTag("pfImpactParameterAK8TagInfos"),
    extSVDeltaRToJet = cms.double(0.8)
)
pfInclusiveSecondaryVertexFinderAK8TagInfos.trackSelection.jetDeltaRMax = cms.double(0.8) # plays no role since using IVF vertices
pfInclusiveSecondaryVertexFinderAK8TagInfos.vertexCuts.maxDeltaRToJetAxis = cms.double(0.8)
