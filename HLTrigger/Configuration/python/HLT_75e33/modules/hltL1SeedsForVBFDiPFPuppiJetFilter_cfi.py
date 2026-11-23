import FWCore.ParameterSet.Config as cms

# L1 seed for VBF paths
# Seed of the default L1 menu (step1_2024):
# pDoublePuppiJet160_35_mass620
# With the v0p0_2026 L1 menu, customizeHLTforL1P2GTv0p0_2026 re-maps it to:
# pDoubleJet_Mass_Min1000
# VBF paths for which v0p0 has dedicated seeds (VBF+MET, VBF+lepton(s), etc.)
# must be accounted for in that customization.
hltL1SeedsForVBFDiPFPuppiJetFilter = cms.EDFilter("HLTP2GTSingleObjectFilter",
    saveTags = cms.bool(True),
    l1GTAlgoBlockTag = cms.InputTag('l1tGTAlgoBlockProducer'),
    minN = cms.uint32(2),
    l1GTAlgos = cms.VPSet(
        cms.PSet(
            name = cms.string('pDoublePuppiJet160_35_mass620'),
            collection = cms.PSet(
                objectType = cms.string('CL2JetsSC4'),
                minPt = cms.double(0),
                maxAbsEta = cms.double(9999.)
            )
        )
    )
)
