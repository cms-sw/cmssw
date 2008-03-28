import FWCore.ParameterSet.Config as cms

#
# producer for hltPhotonEcalIsolFilter
#
hltPhotonEcalIsolFilter = cms.EDFilter("HLTEgammaEcalIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonEcalNonIsol"),
    ncandcut = cms.int32(1),
    ecalisolcut = cms.double(1.5),
    isoTag = cms.InputTag("hltPhotonEcalIsol"),
    candTag = cms.InputTag("hltPhotonEtFilter")
)


