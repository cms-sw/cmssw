import FWCore.ParameterSet.Config as cms

# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltZMMPA = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltZMMPA.HLTPaths = ["HLT_PAL3Mu15_v*"]
hltZMMPA.throw = False
hltZMMPA.andOr = True

# selection of valid vertex
primaryVertexFilterForZMM = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# selection of dimuons with mass in Z range
muonSelectorForZMM = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("(isTrackerMuon && isGlobalMuon) && pt > 25."),
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
    hltZMMPA *
    primaryVertexFilterForZMM *
    muonSelectorForZMM *
    muonFilterForZMM *
    dimuonMassCutForZMM *
    dimuonMassCutFilterForZMM
    )
