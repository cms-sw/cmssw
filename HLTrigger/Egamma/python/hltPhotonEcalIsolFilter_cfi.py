import FWCore.ParameterSet.Config as cms

hltPhotonEcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonEcalNonIsol"),
    ncandcut = cms.int32(1),
    ecalIsoOverEt2Cut = cms.double(0.0),
    ecalisolcut = cms.double(1.5),
    isoTag = cms.InputTag("hltPhotonEcalIsol"),
    candTag = cms.InputTag("hltPhotonEtFilter"),
    ecalIsoOverEtCut = cms.double(0.05)
)


