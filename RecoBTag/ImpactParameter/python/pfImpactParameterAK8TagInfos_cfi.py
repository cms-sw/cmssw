import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.pfImpactParameterTagInfos_cfi import *

pfImpactParameterAK8TagInfos = pfImpactParameterTagInfos.clone(
    computeProbabilities = False,
    computeGhostTrack    = False,
    jets                 = "ak8PFJetsPuppi",
    maxDeltaR            = 0.8
)
