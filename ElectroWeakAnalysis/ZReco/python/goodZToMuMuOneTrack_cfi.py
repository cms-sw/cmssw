import FWCore.ParameterSet.Config as cms

goodZToMuMuOneTrack = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('mass > 0'),
    decay = cms.string('goodMuons@+ goodTracks@-')
)


