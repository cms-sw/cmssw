import FWCore.ParameterSet.Config as cms

hltEG108EtUnseededFilter = cms.EDFilter("HLTEgammaEtFilter",
    etcutEB = cms.double(108.0),
    etcutEE = cms.double(9999999.0),
    inputTag = cms.InputTag("hltEgammaCandidatesWrapperUnseeded"),
    l1EGCand = cms.InputTag("hltEgammaCandidatesUnseeded"),
    ncandcut = cms.int32(1),
    saveTags = cms.bool(True)
)
