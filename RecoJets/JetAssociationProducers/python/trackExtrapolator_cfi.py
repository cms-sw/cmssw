import FWCore.ParameterSet.Config as cms

trackExtrapolator = cms.EDProducer("TrackExtrapolator",
                                   trackSrc = cms.InputTag("generalTracks"),
                                   radii = cms.vdouble( 129.0 ), 
                                   trackQuality = cms.string('goodIterative')
                                   )

