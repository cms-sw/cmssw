import FWCore.ParameterSet.Config as cms

# e/gamma candidate with ET > 22 GeV.
hltEG22EtL1SeededFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcutEB = cms.double(22.0),
    etcutEE = cms.double(22.0),
    inputTag = cms.InputTag("hltEgammaCandidatesWrapperL1Seeded"),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    ncandcut = cms.int32(1),
    saveTags = cms.bool(True)
)
