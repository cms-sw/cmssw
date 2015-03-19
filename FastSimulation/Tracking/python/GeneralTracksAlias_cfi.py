import FWCore.ParameterSet.Config as cms

generalTracksAliasInfo = cms.PSet(
    key = cms.string("mix"),
    value = cms.VPSet( cms.PSet(type=cms.string('recoTracks'),
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

