import FWCore.ParameterSet.Config as cms

# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltZMMPA = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltZMMPA.HLTPaths = ["HLT_PAL3Mu15_v*"]
hltZMMPA.throw = False
hltZMMPA.andOr = True

# selection of valid vertex
primaryVertexFilterForZMMPA = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# selection of dimuons with mass in Z range
muonSelectorForZMMPA = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("(isTrackerMuon && isGlobalMuon) && pt > 25."),
    filter = cms.bool(True)
    )

muonFilterForZMMPA = cms.EDFilter("MuonCountFilter",
    src = cms.InputTag("muonSelectorForZMMPA"),
    minNumber = cms.uint32(2)
    )

dimuonMassCutForZMMPA = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string(' 80 < mass < 110'),
    decay = cms.string("muonSelectorForZMMPA@+ muonSelectorForZMMPA@-")
    )

dimuonMassCutFilterForZMMPA = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuonMassCutForZMMPA"),
    minNumber = cms.uint32(1)
    )

# Z->mumu skim sequence
zMMPASkimSequence = cms.Sequence(
    hltZMMPA *
    primaryVertexFilterForZMMPA *
    muonSelectorForZMMPA *
    muonFilterForZMMPA *
    dimuonMassCutForZMMPA *
    dimuonMassCutFilterForZMMPA
    )
