import FWCore.ParameterSet.Config as cms

electronTrajectoryCleanerBySharedHits = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('electronTrajectoryCleanerBySharedHits'),
    ComponentType = cms.string('TrajectoryCleanerBySharedHits'),
    MissingHitPenalty = cms.double(0.0),
    ValidHitBonus = cms.double(1000.0),
    allowSharedFirstHit = cms.bool(True),
    fractionShared = cms.double(0.19)
)
