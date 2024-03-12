import FWCore.ParameterSet.Config as cms

detachedQuadStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet(
            refToPSet_ = cms.string('detachedQuadStepTrajectoryFilterBase')
        ),
        cms.PSet(
            refToPSet_ = cms.string('ClusterShapeTrajectoryFilter')
        )
    )
)
# foo bar baz
# M2qz1t2gHI1wG
