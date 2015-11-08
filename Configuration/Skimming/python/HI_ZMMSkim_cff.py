import FWCore.ParameterSet.Config as cms

# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltZMMHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltZMMHI.HLTPaths = ["HLT_HIL1DoubleMu10_v*","HLT_HIL3Mu20_2HF_v*","HLT_HIL3Mu20_2HF0_v*"]
hltZMMHI.throw = False
hltZMMHI.andOr = True

# selection of valid vertex
primaryVertexFilterForZMM = cms.EDFilter("VertexSelector",
    src = cms.InputTag("hiSelectedVertex"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# selection of dimuons (at least STA+STA) with mass in Z range
muonSelectorForZMM = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("(isTrackerMuon && isGlobalMuon) && pt > 10."),
    filter = cms.bool(True)
    )

muonFilterForZMM = cms.EDFilter("MuonCountFilter",
    src = cms.InputTag("muonSelectorForZMM"),
    minNumber = cms.uint32(2)
    )

dimuonMassCutForZMM = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string(' 80 < mass < 110'),
    decay = cms.string("muonSelectorForZMM@+ muonSelectorForZMM@-")
    )

dimuonMassCutFilterForZMM = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuonMassCutForZMM"),
    minNumber = cms.uint32(1)
    )

# Z->mumu skim sequence
zMMSkimSequence = cms.Sequence(
    hltZMMHI *
    primaryVertexFilterForZMM *
    muonSelectorForZMM *
    muonFilterForZMM *
    dimuonMassCutForZMM *
    dimuonMassCutFilterForZMM
    )

