import FWCore.ParameterSet.Config as cms

# HLT jet trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltJet150 = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltJet150.HLTPaths = ["HLT_AK4CaloJet150_v*"]
hltJet150.throw = False
hltJet150.andOr = True

# selection of valid vertex
primaryVertexFilterForHighPtJets = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVerticesWithBS"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

highPtJetSkimSequence = cms.Sequence(
        hltJet150*
        primaryVertexFilterForHighPtJets 
)
