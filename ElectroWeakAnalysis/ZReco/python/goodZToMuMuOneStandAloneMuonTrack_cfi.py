import FWCore.ParameterSet.Config as cms

goodZToMuMuOneStandAloneMuonTrack = cms.EDFilter("CandViewShallowCloneCombiner",
    cut = cms.string('mass > 20'),
    decay = cms.string('goodMuons@+ goodStandAloneMuonTracks@-')
)


