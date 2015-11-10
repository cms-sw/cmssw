import FWCore.ParameterSet.Config as cms

# HLT jet trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltTrackHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltTrackHI.HLTPaths = ["HLT_HIFullTrack45_v*"]
hltTrackHI.throw = False
hltTrackHI.andOr = True

# selection of valid vertex                                                                                                                                             
primaryVertexFilterForTrack = cms.EDFilter("VertexSelector",
    src = cms.InputTag("hiSelectedVertex"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"),
    filter = cms.bool(True),   # otherwise it won't filter the events                                                                                                    
    )

singleTrackSkimSequence = cms.Sequence(
    primaryVertexFilterForTrack 
    *hltTrackHI
)
