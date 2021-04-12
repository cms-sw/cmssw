import FWCore.ParameterSet.Config as cms

TrackDistanceValueMapProducer = cms.EDProducer('TrackDistanceValueMapProducer',
                                               muonTracks = cms.InputTag('muonTracks'), # input muon tracks
                                               allTracks = cms.InputTag('generalTracks')   # input generalTracks
                                               )
