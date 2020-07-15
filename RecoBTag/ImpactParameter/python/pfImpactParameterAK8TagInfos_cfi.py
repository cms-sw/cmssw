import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.pfImpactParameterTagInfos_cfi import *

pfImpactParameterAK8TagInfos = pfImpactParameterTagInfos.clone(
    computeProbabilities = cms.bool(False),
    computeGhostTrack = cms.bool(False),
    jets = cms.InputTag("ak8PFJetsPuppi"),
    maxDeltaR = cms.double(0.8)
)
