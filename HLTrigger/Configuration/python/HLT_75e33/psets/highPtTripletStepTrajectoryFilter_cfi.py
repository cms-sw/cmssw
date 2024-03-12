import FWCore.ParameterSet.Config as cms

highPtTripletStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet(
            refToPSet_ = cms.string('highPtTripletStepTrajectoryFilterBase')
        ),
        cms.PSet(
            refToPSet_ = cms.string('ClusterShapeTrajectoryFilter')
        )
    )
)