import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonHighPtTripletStepTrajectoryCleanerBySharedHits = cms.ESProducer("TrajectoryCleanerESProducer",
    ComponentName = cms.string('hltPhase2L3MuonHighPtTripletStepTrajectoryCleanerBySharedHits'),
    ComponentType = cms.string('TrajectoryCleanerBySharedHits'),
    MissingHitPenalty = cms.double(20.0),
    ValidHitBonus = cms.double(5.0),
    allowSharedFirstHit = cms.bool(True),
    fractionShared = cms.double(0.16)
)
