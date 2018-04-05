import FWCore.ParameterSet.Config as cms

import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
TrajectoryFilterForConversions = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    chargeSignificance = cms.double(-1.0),
    minPt = cms.double(0.9),
    minHitsMinPt = cms.int32(-1),
    ComponentType = cms.string('CkfBaseTrajectoryFilter'),
    maxLostHits = cms.int32(1),
    maxNumberOfHits = cms.int32(-1),
    maxConsecLostHits = cms.int32(1),
    nSigmaMinPt = cms.double(5.0),
    minimumNumberOfHits = cms.int32(3),
    maxCCCLostHits = cms.int32(9999),
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutNone'))
)

