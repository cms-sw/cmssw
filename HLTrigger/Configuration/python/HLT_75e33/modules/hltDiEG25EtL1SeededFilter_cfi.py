import FWCore.ParameterSet.Config as cms

hltDiEG25EtL1SeededFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcutEB = cms.double(25.0),
    etcutEE = cms.double(25.0),
    inputTag = cms.InputTag("hltEgammaCandidatesWrapperL1Seeded"),
    l1EGCand = cms.InputTag("hltEgammaCandidatesL1Seeded"),
    ncandcut = cms.int32(2),
    saveTags = cms.bool(True)
)
