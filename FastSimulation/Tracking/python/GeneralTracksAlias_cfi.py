import FWCore.ParameterSet.Config as cms

generalTracks = cms.EDAlias(
    mix = cms.VPSet( cms.PSet(type=cms.string('recoTracks'),
                              fromProductInstance = cms.string('generalTracks'),
                              toProductInstance = cms.string('') ),
                     cms.PSet(type=cms.string('recoTrackExtras'),
                              fromProductInstance = cms.string('generalTracks'),
                              toProductInstance = cms.string('') ),
                     cms.PSet(type=cms.string('TrackingRecHitsOwned'),
                              fromProductInstance = cms.string('generalTracks'),
                              toProductInstance = cms.string('') ),
                     cms.PSet(type=cms.string('floatedmValueMap'),
                              fromProductInstance = cms.string('generalTracksMVAVals'),
                              toProductInstance = cms.string('MVAVals') ) )
    )
