import FWCore.ParameterSet.Config as cms

muonRoadTrajectoryBuilderESProducer = cms.ESProducer("MuonRoadTrajectoryBuilderESProducer",
    minNumberOfHitOnCandidate = cms.uint32(4),
    maxTrajectories = cms.uint32(30),
    #category: MuonRoadTrajectoryBuilder
    ComponentName = cms.string('muonRoadTrajectoryBuilder'),
    outputAllTraj = cms.bool(True),
    numberOfHitPerModuleThreshold = cms.vuint32(3, 1),
    measurementTrackerName = cms.string(''),
    dynamicMaxNumberOfHitPerModule = cms.bool(True),
    maxChi2Road = cms.double(40.0),
    maxChi2Hit = cms.double(40.0),
    propagatorName = cms.string('SteppingHelixPropagatorAny'),
    numberOfHitPerModule = cms.uint32(1000), ##no cut then

    maxTrajectoriesThreshold = cms.vuint32(10, 25)
)



