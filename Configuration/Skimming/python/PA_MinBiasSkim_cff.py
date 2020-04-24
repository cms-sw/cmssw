import FWCore.ParameterSet.Config as cms

# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltMinBiasHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltMinBiasHI.HLTPaths = ["HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_ForSkim_v*"]
hltMinBiasHI.throw = False
hltMinBiasHI.andOr = True

# selection of valid vertex
primaryVertexFilterForMinBias = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# MinBias skim sequence
minBiasSkimSequence = cms.Sequence(
    hltMinBiasHI *
    primaryVertexFilterForMinBias
)
