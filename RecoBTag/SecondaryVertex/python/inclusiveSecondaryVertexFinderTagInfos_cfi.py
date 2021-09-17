import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.secondaryVertexTagInfos_cfi import *

inclusiveSecondaryVertexFinderTagInfos = secondaryVertexTagInfos.clone(
# use external SV collection made from IVF
    extSVCollection  = 'inclusiveSecondaryVertices',
    extSVDeltaRToJet = 0.3,
    useExternalSV    = True,
    vertexCuts       = dict(fracPV = 0.79, ## 4 out of 5 is discarded
                            distSig2dMin = 2.0)
)
