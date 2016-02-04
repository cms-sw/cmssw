import FWCore.ParameterSet.Config as cms

dimuonsOneTrack = cms.EDProducer("CandViewShallowCloneCombiner",
    checkCharge = cms.bool(True),
    cut = cms.string('mass > 20'),
#    decay = cms.string('selectedLayer1Muons@+ goodTracks@-')
#    decay = cms.string('selectedLayer1Muons@+ selectedLayer1TrackCands@-')
    decay = cms.string('selectedPatMuonsTriggerMatch@+ selectedPatTracks@-')
)


