import FWCore.ParameterSet.Config as cms

trackExtrapolator = cms.EDProducer("TrackExtrapolator",
                                   trackSrc = cms.InputTag("generalTracks"),
                                   trackQuality = cms.string('goodIterative')
                                   )

