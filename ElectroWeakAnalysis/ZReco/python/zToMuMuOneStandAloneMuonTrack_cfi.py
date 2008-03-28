import FWCore.ParameterSet.Config as cms

zToMuMuOneStandAloneMuonTrack = cms.EDFilter("MassMinCandShallowCloneCombiner",
    massMin = cms.double(20.0),
    decay = cms.string('allMuons@+ allStandAloneMuonTracks@-')
)


