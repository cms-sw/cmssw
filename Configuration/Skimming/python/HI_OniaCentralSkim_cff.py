import FWCore.ParameterSet.Config as cms

# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltCentralOniaMMHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltCentralOniaMMHI.HLTPaths = ["HLT_HIL3DoubleMu0_Cent30_OS_m2p5to4p5_v*","HLT_HIL3DoubleMu0_Cent30_OS_m7to14_v*"] 
hltCentralOniaMMHI.throw = False
hltCentralOniaMMHI.andOr = True

# selection of valid vertex
primaryVertexFilterForOniaMMCentral = cms.EDFilter("VertexSelector",
    src = cms.InputTag("hiSelectedVertex"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# selection of dimuons with mass in Jpsi or 
muonSelectorForOniaMMCentral = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("(isTrackerMuon && isGlobalMuon) && pt > 1.5"),
    filter = cms.bool(True)
    )

muonFilterForOniaMMCentral = cms.EDFilter("MuonCountFilter",
    src = cms.InputTag("muonSelectorForOniaMMCentral"),
    minNumber = cms.uint32(2)
    )

# opposite charge only 
dimuonMassCutForOniaMMCentral = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string(' (2.6 < mass < 3.5) || (7.0 < mass < 14.0)'),
    decay = cms.string("muonSelectorForOniaMMCentral@+ muonSelectorForOniaMMCentral@-")
    )

dimuonMassCutFilterForOniaMMCentral = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuonMassCutForOniaMMCentral"),
    minNumber = cms.uint32(1)
    )

# onia skim sequence
oniaCentralSkimSequence = cms.Sequence(
    hltCentralOniaMMHI *
    primaryVertexFilterForOniaMMCentral *
    muonSelectorForOniaMMCentral *
    muonFilterForOniaMMCentral *
    dimuonMassCutForOniaMMCentral *
    dimuonMassCutFilterForOniaMMCentral
    )
