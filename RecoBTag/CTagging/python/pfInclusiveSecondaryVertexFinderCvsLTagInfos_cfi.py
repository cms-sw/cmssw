import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.pfSecondaryVertexTagInfos_cfi import *

# use external SV collection made from IVF
pfInclusiveSecondaryVertexFinderCvsLTagInfos = pfSecondaryVertexTagInfos.clone(
    extSVCollection  = 'inclusiveCandidateSecondaryVerticesCvsL',
    extSVDeltaRToJet = 0.3,
    useExternalSV    = True,
    vertexCuts = dict(fracPV = 0.79, ## 4 out of 5 is discarded
    		      distSig2dMin = 1.5 # relaxed w.r.t taginfos used in btagger
		     )
)
