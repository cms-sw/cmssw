import FWCore.ParameterSet.Config as cms

# HLT jet trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltHIJet150 = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltHIJet150.HLTPaths = ["HLT_HIPuAK4CaloJet150_Eta5p1_v*"]
hltHIJet150.throw = False
hltHIJet150.andOr = True

# selection of valid vertex
primaryVertexFilterForHighPtJets = cms.EDFilter("VertexSelector",
    src = cms.InputTag("hiSelectedVertex"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

highPtJetSkimSequence = cms.Sequence(
        hltHIJet150*
        primaryVertexFilterForHighPtJets 
)


