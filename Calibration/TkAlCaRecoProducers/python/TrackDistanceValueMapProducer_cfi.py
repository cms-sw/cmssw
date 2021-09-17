import FWCore.ParameterSet.Config as cms

TrackDistanceValueMapProducer = cms.EDProducer('TrackDistanceValueMapProducer',
                                               muonTracks = cms.InputTag('muonTracks'),   # input muon tracks
                                               allTracks = cms.InputTag('generalTracks'), # input generalTracks
                                               saveUpToNthClosest =  cms.uint32(1)        # save only closest track
                                               )
