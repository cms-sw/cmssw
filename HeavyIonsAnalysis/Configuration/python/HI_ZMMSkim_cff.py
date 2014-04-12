import FWCore.ParameterSet.Config as cms

# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltZMMHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltZMMHI.HLTPaths = ["HLT_HIL2DoubleMu3"]
hltZMMHI.throw = False
hltZMMHI.andOr = True

# selection of valid vertex
primaryVertexFilterForZMM = cms.EDFilter("VertexSelector",
    src = cms.InputTag("hiSelectedVertex"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# selection of dimuons (at least STA+STA) with mass in Z range
muonSelector = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("(isStandAloneMuon || isGlobalMuon) && pt > 1."),
    filter = cms.bool(True)
    )

muonFilter = cms.EDFilter("MuonCountFilter",
    src = cms.InputTag("muonSelector"),
    minNumber = cms.uint32(1)
    )

dimuonMassCut = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string(' 60 < mass < 120'),
    decay = cms.string("muonSelector@+ muonSelector@-")
    )

dimuonMassCutFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuonMassCut"),
    minNumber = cms.uint32(1)
    )

# Z->mumu skim sequence
zMMSkimSequence = cms.Sequence(
    hltZMMHI *
    primaryVertexFilterForZMM *
    muonSelector *
    muonFilter *
    dimuonMassCut *
    dimuonMassCutFilter
    )
