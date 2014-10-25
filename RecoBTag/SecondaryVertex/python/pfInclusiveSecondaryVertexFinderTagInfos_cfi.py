import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.pfSecondaryVertexTagInfos_cfi import *

pfInclusiveSecondaryVertexFinderTagInfos = pfSecondaryVertexTagInfos.clone()

# use external SV collection made from IVF
pfInclusiveSecondaryVertexFinderTagInfos.extSVCollection     = cms.InputTag('inclusiveCandidateSecondaryVertices')
pfInclusiveSecondaryVertexFinderTagInfos.extSVDeltaRToJet    = cms.double(0.3)
pfInclusiveSecondaryVertexFinderTagInfos.useExternalSV = cms.bool(True)
pfInclusiveSecondaryVertexFinderTagInfos.vertexCuts.fracPV = 0.79 ## 4 out of 5 is discarded
pfInclusiveSecondaryVertexFinderTagInfos.vertexCuts.distSig2dMin = 2.0


