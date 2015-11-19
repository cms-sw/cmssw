import FWCore.ParameterSet.Config as cms

# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltMinBiasHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltMinBiasHI.HLTPaths = ["HLT_HIL1MinimumBiasHF2AND_v*"]
hltMinBiasHI.throw = False
hltMinBiasHI.andOr = True

# selection of valid vertex
primaryVertexFilterForMinBias = cms.EDFilter("VertexSelector",
    src = cms.InputTag("hiSelectedVertex"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# MinBias skim sequence
minBiasSkimSequence = cms.Sequence(
    hltMinBiasHI *
    primaryVertexFilterForMinBias
    )
