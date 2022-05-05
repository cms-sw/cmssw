import FWCore.ParameterSet.Config as cms

hltEG187EtUnseededFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcutEB = cms.double(187.0),
    etcutEE = cms.double(187.0),
    inputTag = cms.InputTag("hltEgammaCandidatesWrapperUnseeded"),
    l1EGCand = cms.InputTag("hltEgammaCandidatesUnseeded"),
    ncandcut = cms.int32(1),
    saveTags = cms.bool(True)
)
