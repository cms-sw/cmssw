import FWCore.ParameterSet.Config as cms

CompositeTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet()
)