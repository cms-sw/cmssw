import FWCore.ParameterSet.Config as cms

ChargeSignificanceTrajectoryFilter_block = cms.PSet(
    ComponentType = cms.string('ChargeSignificanceTrajectoryFilter'),
    chargeSignificance = cms.double(-1.0)
)