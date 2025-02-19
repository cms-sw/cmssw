import FWCore.ParameterSet.Config as cms

TrackerSeedCleanerCommon = cms.PSet(
    TrackerSeedCleaner = cms.PSet(
        cleanerFromSharedHits = cms.bool(True),
        # should be true only for tests
        ptCleaner = cms.bool(True),
        TTRHBuilder = cms.string('WithTrackAngle'),
        beamSpot = cms.InputTag("hltOfflineBeamSpot"),
        directionCleaner = cms.bool(True)
    )
)


