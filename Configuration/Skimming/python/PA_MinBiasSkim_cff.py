import FWCore.ParameterSet.Config as cms

# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltMinBiasPA = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltMinBiasPA.HLTPaths = ["HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_ForSkim_v*"]
hltMinBiasPA.throw = False
hltMinBiasPA.andOr = True

# selection of valid vertex
primaryVertexFilterForMinBiasPA = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# MinBias skim sequence
minBiasPASkimSequence = cms.Sequence(
    hltMinBiasPA *
    primaryVertexFilterForMinBiasPA
)
