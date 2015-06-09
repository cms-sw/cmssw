import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.pfImpactParameterTagInfos_cfi import *

pfImpactParameterTagInfosAK8 = pfImpactParameterTagInfos.clone(
    computeProbabilities = cms.bool(False),
    computeGhostTrack = cms.bool(False),
    jets = cms.InputTag("ak8PFJetsCHS"),
    maxDeltaR = cms.double(0.8)
)
