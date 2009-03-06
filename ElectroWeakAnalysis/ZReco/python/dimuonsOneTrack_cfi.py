import FWCore.ParameterSet.Config as cms

dimuonsOneTrack = cms.EDFilter("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(False),
    cut = cms.string('mass > 0'),
    #  string decay = "goodMuons@+ goodTracks@-"
    decay = cms.string('selectedLayer1Muons@+ goodTracks@-')
)


