import FWCore.ParameterSet.Config as cms

#
# producer for hltSinglePhotonTrackIsolFilter
#
hltPhotonTrackIsolFilter = cms.EDFilter("HLTPhotonTrackIsolFilter",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltPhotonTrackIsol"),
    candTag = cms.InputTag("hltPhotonHcalIsolFilter"),
    numtrackisolcut = cms.double(1.0)
)


