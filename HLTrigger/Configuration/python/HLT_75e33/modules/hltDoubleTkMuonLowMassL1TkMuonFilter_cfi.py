import FWCore.ParameterSet.Config as cms

# L1 gate of the HLT_DoubleMu4_3_{LowMass,Bs,Jpsi}_FromL1TkMuon paths: two GMTTkMuons from
# one of the low-mass opposite-sign double-muon seeds. The seeds are named as in the step1_2024
# L1 menu, like all the other HLT_75e33 gates; customizeHLTforL1P2GTv0p0_2026 re-wires them to
# pDoubleTkMu0er1p4_SQ_OS_dR_Max1p4 and pDoubleTkMu4er2p0_SQ_OS_dR_Max1p6 for the v0p0_2026 menu.
hltDoubleTkMuonLowMassL1TkMuonFilter = cms.EDFilter("HLTP2GTSingleObjectFilter",
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
            name = cms.string("pDoubleTkMuon_4_4_OS_Dr1p2"),
            collection = cms.PSet(
                objectType = cms.string("GMTTkMuons"),
                minPt = cms.double(0),
                maxAbsEta = cms.double(9999.)
            )
        )
    )
)
