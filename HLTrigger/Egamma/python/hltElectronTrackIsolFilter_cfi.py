import FWCore.ParameterSet.Config as cms

hltElectronTrackIsolFilter = cms.EDFilter("HLTElectronTrackIsolFilterRegional",
    doIsolated = cms.bool(True),
    nonIsoTag = cms.InputTag("hltPhotonNonIsoTrackIsol"),
    pttrackisolcut = cms.double(0.06),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltElectronTrackIsol"),
    candTag = cms.InputTag("hltElectronEoverpFilter")
)


