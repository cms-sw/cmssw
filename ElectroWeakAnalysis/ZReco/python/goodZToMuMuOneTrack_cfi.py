import FWCore.ParameterSet.Config as cms

goodZToMuMuOneTrack = cms.EDFilter("CandViewShallowCloneCombiner",
    cut = cms.string('mass > 20'),
    decay = cms.string('goodMuons@+ goodTracks@-')
)


