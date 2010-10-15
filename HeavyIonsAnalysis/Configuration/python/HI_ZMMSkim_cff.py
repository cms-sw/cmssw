import FWCore.ParameterSet.Config as cms

# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltZMMHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltZMMHI.HLTPaths = ["HLT_L1DoubleMuOpen"]
hltZMMHI.throw = False
hltZMMHI.andOr = True

muonSelector = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("(isStandAloneMuon || isGlobalMuon) && pt > 1."),
    filter = cms.bool(True)
    )

muonFilter = cms.EDFilter("MuonCountFilter",
    src = cms.InputTag("muonSelector"),
    minNumber = cms.uint32(1)
    )

dimuonsMassCut = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string(' mass > 60 & mass < 120 & charge=0'),
    decay = cms.string("muonSelector@+ muonSelector@-")
    )

dimuonsMassCutFilter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuonsMassCut"),
    minNumber = cms.uint32(1)
    )

# Z->mumu skim sequence
zMMSkimSequence = cms.Sequence(
    hltZMMHI *
    muonSelector *
    muonFilter *
    dimuonsMassCut *
    dimuonsMassCutFilter
    )
