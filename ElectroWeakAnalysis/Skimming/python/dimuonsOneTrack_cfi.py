import FWCore.ParameterSet.Config as cms

dimuonsOneTrack = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('mass > 0'),
#    decay = cms.string('selectedLayer1Muons@+ goodTracks@-')
    decay = cms.string('selectedLayer1Muons@+ selectedLayer1TrackCands@-')
#    decay = cms.string('selectedLayer1MuonsTriggerMatch@+ selectedLayer1TrackCands@-')
)


