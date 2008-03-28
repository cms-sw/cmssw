import FWCore.ParameterSet.Config as cms

zToMuMuOneTrack = cms.EDFilter("MassMinCandShallowCloneCombiner",
    massMin = cms.double(20.0),
    decay = cms.string('allMuons@+ allTracks@-')
)


