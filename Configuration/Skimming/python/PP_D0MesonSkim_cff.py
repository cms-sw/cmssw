import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
hltDmeson60 = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltDmeson60.HLTPaths = ["HLT_DmesonPPTrackingGlobal_Dpt60_v*"]
hltDmeson60.throw = False
hltDmeson60.andOr = True

# selection of valid vertex                                                                                                                                             
primaryVertexFilterForD0Meson = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVerticesWithBS"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"),
    filter = cms.bool(True),   # otherwise it won't filter the events                                                                                         
    )

d0MesonSkimSequence = cms.Sequence(
    primaryVertexFilterForD0Meson 
    *hltDmeson60
)
