import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.pfSecondaryVertexTagInfos_cfi import *

# use external SV collection made from IVF
pfInclusiveSecondaryVertexFinderCvsLTagInfos = pfSecondaryVertexTagInfos.clone(
   extSVCollection  = cms.InputTag('inclusiveCandidateSecondaryVerticesCvsL'),
   extSVDeltaRToJet = cms.double(0.3),
   useExternalSV    = cms.bool(True)
)
pfInclusiveSecondaryVertexFinderCvsLTagInfos.vertexCuts.fracPV = 0.79 ## 4 out of 5 is discarded
pfInclusiveSecondaryVertexFinderCvsLTagInfos.vertexCuts.distSig2dMin = 1.5 # relaxed w.r.t taginfos used in btagger


