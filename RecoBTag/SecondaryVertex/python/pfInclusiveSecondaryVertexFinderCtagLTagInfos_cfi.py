import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.pfSecondaryVertexTagInfos_cfi import *

pfInclusiveSecondaryVertexFinderCtagLTagInfos = pfSecondaryVertexTagInfos.clone()

# use external SV collection made from IVF
pfInclusiveSecondaryVertexFinderCtagLTagInfos.extSVCollection     = cms.InputTag('inclusiveCandidateSecondaryVerticesCtagL')
pfInclusiveSecondaryVertexFinderCtagLTagInfos.extSVDeltaRToJet    = cms.double(0.3)
pfInclusiveSecondaryVertexFinderCtagLTagInfos.useExternalSV = cms.bool(True)
pfInclusiveSecondaryVertexFinderCtagLTagInfos.vertexCuts.fracPV = 0.79 ## 4 out of 5 is discarded
pfInclusiveSecondaryVertexFinderCtagLTagInfos.vertexCuts.distSig2dMin = 1.5 # relaxed w.r.t taginfos used in btagger


