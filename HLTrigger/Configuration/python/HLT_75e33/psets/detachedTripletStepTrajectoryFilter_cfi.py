import FWCore.ParameterSet.Config as cms

detachedTripletStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(cms.PSet(
        refToPSet_ = cms.string('detachedTripletStepTrajectoryFilterBase')
    ))
)
# foo bar baz
# YVbCL76d5mKT2
# 9iVqWmYjFNSEP
