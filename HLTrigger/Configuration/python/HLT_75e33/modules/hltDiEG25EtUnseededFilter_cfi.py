import FWCore.ParameterSet.Config as cms

hltDiEG25EtUnseededFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcutEB = cms.double(25.0),
    etcutEE = cms.double(25.0),
    inputTag = cms.InputTag("hltEgammaCandidatesWrapperUnseeded"),
    l1EGCand = cms.InputTag("hltEgammaCandidatesUnseeded"),
    ncandcut = cms.int32(2),
    saveTags = cms.bool(True)
)
