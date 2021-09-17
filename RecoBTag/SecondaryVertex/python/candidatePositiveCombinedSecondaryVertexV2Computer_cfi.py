import FWCore.ParameterSet.Config as cms

from RecoBTag.SecondaryVertex.candidateCombinedSecondaryVertexV2Computer_cfi import *

candidatePositiveCombinedSecondaryVertexV2Computer = candidateCombinedSecondaryVertexV2Computer.clone(
    trackSelection = dict(sip3dSigMin = 0),
    trackPseudoSelection = dict(sip3dSigMin = 0)
)
