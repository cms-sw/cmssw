import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.pfImpactParameterTagInfos_cfi import *

pfImpactParameterCA15TagInfos = pfImpactParameterTagInfos.clone(
    computeProbabilities = False,
    computeGhostTrack    = False,
    jets                 = "ca15PFJetsCHS",
    maxDeltaR            = 1.5
)
# foo bar baz
# P7ZQf05l03Qno
