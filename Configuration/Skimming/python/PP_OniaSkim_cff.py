import FWCore.ParameterSet.Config as cms

# HLT dimuon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltOniaMM = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltOniaMM.HLTPaths = ["HLT_HIL1DoubleMu0_v*"] 
hltOniaMM.throw = False
hltOniaMM.andOr = True

# selection of valid vertex
primaryVertexFilterForOniaMM = cms.EDFilter("VertexSelector",
    src = cms.InputTag("offlinePrimaryVerticesWithBS"),
    cut = cms.string("!isFake && abs(z) <= 25 && position.Rho <= 2"), 
    filter = cms.bool(True),   # otherwise it won't filter the events
    )

# selection of dimuons with mass in Jpsi or 
muonSelectorForOniaMM = cms.EDFilter("MuonSelector",
    src = cms.InputTag("muons"),
    cut = cms.string("(isTrackerMuon && isGlobalMuon) && pt > 1.5"),
    filter = cms.bool(True)
    )

muonFilterForOniaMM = cms.EDFilter("MuonCountFilter",
    src = cms.InputTag("muonSelectorForOniaMM"),
    minNumber = cms.uint32(2)
    )

# opposite charge only 
dimuonMassCutForOniaMM = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string(' (2.6 < mass < 3.5) || (7.0 < mass < 14.0)'),
    decay = cms.string("muonSelectorForOniaMM@+ muonSelectorForOniaMM@-")
    )

dimuonMassCutFilterForOniaMM = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("dimuonMassCutForOniaMM"),
    minNumber = cms.uint32(1)
    )

# onia skim sequence
oniaSkimSequence = cms.Sequence(
    hltOniaMM *
    primaryVertexFilterForOniaMM *
    muonSelectorForOniaMM *
    muonFilterForOniaMM *
    dimuonMassCutForOniaMM *
    dimuonMassCutFilterForOniaMM
    )
