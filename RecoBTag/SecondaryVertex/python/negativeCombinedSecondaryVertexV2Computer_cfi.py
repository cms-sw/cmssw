import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.combinedSecondaryVertexV2Computer_cfi import *

negativeCombinedSecondaryVertexV2Computer = combinedSecondaryVertexV2Computer.clone(
    vertexFlip = True,
    trackFlip  = True,
    trackSelection = dict(sip3dSigMax = 0),
    trackPseudoSelection = dict(sip3dSigMax = 0,
                                sip2dSigMin = -99999.9,
                                sip2dSigMax = -2.0)
)
