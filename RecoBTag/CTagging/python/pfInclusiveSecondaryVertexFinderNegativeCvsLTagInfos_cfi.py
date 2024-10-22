import FWCore.ParameterSet.Config as cms

from RecoBTag.CTagging.pfInclusiveSecondaryVertexFinderCvsLTagInfos_cfi import *

pfInclusiveSecondaryVertexFinderNegativeCvsLTagInfos = pfInclusiveSecondaryVertexFinderCvsLTagInfos.clone(
    extSVDeltaRToJet = -0.3,
    vertexCuts = dict(distVal2dMin = -2.5,
		      distVal2dMax = -0.01,
		      distSig2dMin = -99999.9,
		      distSig2dMax = -1.5,
		      maxDeltaRToJetAxis = -0.5)
)
