import FWCore.ParameterSet.Config as cms

# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltZEEHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltZEEHI.HLTPaths = ["HLT_HIDoublePhoton15_Eta1p5_Mass50_1000_R9HECut_v*"]
hltZEEHI.throw = False
hltZEEHI.andOr = True

# selection of valid vertex
primaryVertexFilterForZEE = cms.EDFilter("VertexSelector",
    src = cms.InputTag("hiSelectedVertex"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# Z->ee skim sequence
zEESkimSequence = cms.Sequence(
    hltZEEHI *
    primaryVertexFilterForZEE
    )
