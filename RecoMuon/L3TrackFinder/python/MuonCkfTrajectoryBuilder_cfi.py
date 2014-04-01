import FWCore.ParameterSet.Config as cms

MuonCkfTrajectoryBuilder = cms.PSet(
    ComponentType = cms.string('MuonCkfTrajectoryBuilder'),
    propagatorAlong = cms.string('PropagatorWithMaterial'),
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('muonCkfTrajectoryFilter')),
    maxCand = cms.int32(5),
    intermediateCleaning = cms.bool(False),
    #would skip the first layer to search for measurement if bare TrajectorySeed
    useSeedLayer = cms.bool(False),
    MeasurementTrackerName = cms.string(''),
    estimator = cms.string('Chi2'),
    TTRHBuilder = cms.string('WithTrackAngle'),
    #propagator used only if useSeedLayer=true
    propagatorProximity = cms.string('SteppingHelixPropagatorAny'),
    updator = cms.string('KFUpdator'),
    alwaysUseInvalidHits = cms.bool(True),
    #would rescale the error to find measurements is failing 
    #1.0 would skip this step completely
    rescaleErrorIfFail = cms.double(1.0),
    propagatorOpposite = cms.string('PropagatorWithMaterialOpposite'),
    lostHitPenalty = cms.double(30.0),
    #this is present in HLT config, appears to be dummy
#    appendToDataLabel = cms.string( "" ),
    #safety cone size
    deltaEta = cms.double( 0.1 ),
    deltaPhi = cms.double( 0.1 )
)



