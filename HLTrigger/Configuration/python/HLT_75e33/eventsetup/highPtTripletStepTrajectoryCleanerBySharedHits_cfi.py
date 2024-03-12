import FWCore.ParameterSet.Config as cms

highPtTripletStepTrajectoryCleanerBySharedHits = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('highPtTripletStepTrajectoryCleanerBySharedHits'),
    ComponentType = cms.string('TrajectoryCleanerBySharedHits'),
    MissingHitPenalty = cms.double(20.0),
    ValidHitBonus = cms.double(5.0),
    allowSharedFirstHit = cms.bool(True),
    fractionShared = cms.double(0.16)
)
# foo bar baz
# nkPumCh1I1bBJ
# NeXHcRcB0Zbcp
