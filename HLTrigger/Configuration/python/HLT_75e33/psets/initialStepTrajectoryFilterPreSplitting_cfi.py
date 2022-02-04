import FWCore.ParameterSet.Config as cms

initialStepTrajectoryFilterPreSplitting = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet(
            refToPSet_ = cms.string('initialStepTrajectoryFilterBasePreSplitting')
        ),
        cms.PSet(
            refToPSet_ = cms.string('initialStepTrajectoryFilterShapePreSplitting')
        )
    )
)
