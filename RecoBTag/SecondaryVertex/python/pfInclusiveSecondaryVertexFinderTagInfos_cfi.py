import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.pfSecondaryVertexTagInfos_cfi import *

pfInclusiveSecondaryVertexFinderTagInfos = pfSecondaryVertexTagInfos.clone(
# use external SV collection made from IVF
    extSVCollection  = 'inclusiveCandidateSecondaryVertices',
    extSVDeltaRToJet = 0.3,
    useExternalSV    = True,
    vertexCuts = dict(fracPV = 0.79, ## 4 out of 5 is discarded
                      distSig2dMin = 2.0)
)
