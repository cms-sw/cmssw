import FWCore.ParameterSet.Config as cms

goodZToMuMuOneStandAloneMuonTrack = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('mass > 0'),
    decay = cms.string('goodMuons@+ goodStandAloneMuonTracks@-')
)


