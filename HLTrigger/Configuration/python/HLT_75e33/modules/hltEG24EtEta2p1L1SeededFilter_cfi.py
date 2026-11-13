import FWCore.ParameterSet.Config as cms

# e/gamma candidate with ET > 24 GeV and |eta| < 2.1.
# For now, thresholds aligned to the v0p0_2026
# L1 seed pDoubleJet_TkEleIso.
hltEG24EtEta2p1L1SeededFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcutEB = cms.double(24.0),
    etcutEE = cms.double(24.0),
    inputTag = cms.InputTag("hltEgammaCandidatesWrapperL1Seeded"),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    maxEtaCut = cms.double(2.1),
    minEtaCut = cms.double(-2.1),
    ncandcut = cms.int32(1),
    saveTags = cms.bool(True)
)
