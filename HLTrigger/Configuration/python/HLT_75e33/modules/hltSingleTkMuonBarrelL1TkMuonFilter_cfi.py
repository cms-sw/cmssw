import FWCore.ParameterSet.Config as cms

# L1 gate of the HLT_Mu9_Barrel_IP6_FromL1TkMuon path: one GMTTkMuon from pSingleTkMu_Barrel
# (Medium TkMuon, |eta| < 0.83), the closest analogue of the Run-3 L1_SingleMu10_SQ14_BMTF seed.
# NB: pSingleTkMu_Barrel exists only in the v0p0_2026 L1 menu (there is no low-pT barrel single
# muon seed in step1_2024), so the path has to run with -s L1P2GT:v0p0_2026 and
# customizeHLTforL1P2GTv0p0_2026 (which leaves this name untouched): with the step1_2024 menu
# HLTP2GTSingleObjectFilter throws, since the algorithm is not in the P2GTAlgoBlockMap.
hltSingleTkMuonBarrelL1TkMuonFilter = cms.EDFilter("HLTP2GTSingleObjectFilter",
    saveTags = cms.bool(True),
    l1GTAlgoBlockTag = cms.InputTag("l1tGTAlgoBlockProducer"),
    minN = cms.uint32(1),
    l1GTAlgos = cms.VPSet(
        cms.PSet(
            name = cms.string("pSingleTkMu_Barrel"),
            collection = cms.PSet(
                objectType = cms.string("GMTTkMuons"),
                minPt = cms.double(0),
                maxAbsEta = cms.double(9999.)
            )
        )
    )
)
