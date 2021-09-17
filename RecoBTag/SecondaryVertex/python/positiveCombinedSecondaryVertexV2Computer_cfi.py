import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.combinedSecondaryVertexV2Computer_cfi import *

positiveCombinedSecondaryVertexV2Computer = combinedSecondaryVertexV2Computer.clone(
    trackSelection = dict(sip3dSigMin = 0),
    trackPseudoSelection = dict(sip3dSigMin = 0)
)
