import FWCore.ParameterSet.Config as cms

# L1 gate of the HLT_Dimuon0_Upsilon_FromL1TkMuon path: two GMTTkMuons from one of the opposite-sign
# double-muon seeds closest to the Run-3 L1_DoubleMu0er2p0_SQ_OS_dEta_Max1p5 OR
# L1_DoubleMu4p5er2p0_SQ_OS_Mass_Min7 OR L1_DoubleMu4p5er2p0_SQ_OS_Mass_7to18 seeds.
# The seeds are named as in the step1_2024 L1 menu, like all the other HLT_75e33 gates;
# customizeHLTforL1P2GTv0p0_2026 re-wires them to pDoubleTkMu0er1p4_SQ_OS_dR_Max1p4 and
# pDoubleTkMu4er2p0_SQ_OS_dR_Max1p6 for the v0p0_2026 menu.
# NB: v0p0_2026 has no double-muon seed with a mass window, and both its seeds above cut at
# dR < 1.4-1.6: the wide-angle muon pairs of low-pT Upsilons are lost at L1 (in a generator-level
# toy, no acceptance below pT(Upsilon) ~ 10 GeV), while step1_2024 keeps them through the 7-18 GeV
# mass-window seed. A mass-window seed in v0p0_2026 (e.g. pDoubleTkMu4p5er2p0_SQ_OS_Mass_7to18)
# is needed to recover them.
hltDoubleTkMuonUpsilonL1TkMuonFilter = cms.EDFilter("HLTP2GTSingleObjectFilter",
    saveTags = cms.bool(True),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    minN = cms.uint32(2),
    l1GTAlgos = cms.VPSet(
        cms.PSet(
            name = cms.string("pDoubleTkMuon_OS_Er1p5_Dr1p4"),
            collection = cms.PSet(
                objectType = cms.string("GMTTkMuons"),
                minPt = cms.double(0),
                maxAbsEta = cms.double(9999.)
            )
        ),
        cms.PSet(
            name = cms.string("pDoubleTkMuon_4p5_4p5_OS_Er2_Mass7to18"),
            collection = cms.PSet(
                objectType = cms.string("GMTTkMuons"),
                minPt = cms.double(0),
                maxAbsEta = cms.double(9999.)
            )
        )
    )
)
